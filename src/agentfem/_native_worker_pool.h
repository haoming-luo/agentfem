/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Instance-owned sleeping workers. No numerical state, retained job buffers,
 * global executor, busy wait or callback into Python on the scheduler path.
 */
#include <unistd.h>
typedef struct native_pool NativePool;
typedef struct { NativePool *pool; int index; } NativeWorker;
struct native_pool {
    PyObject_HEAD
    pthread_mutex_t mutex;
    pthread_cond_t ready, done;
    pthread_t threads[16];
    NativeWorker arguments[16];
    int count, started, stop, closed, busy, active, remaining;
    unsigned long generation;
    pid_t process;
    void *jobs;
    size_t stride;
    void *(*execute)(void *);
};

static void *native_pool_worker(void *raw) {
    NativeWorker *worker=raw; NativePool *pool=worker->pool;
    unsigned long generation=0;
    pthread_mutex_lock(&pool->mutex);
    for(;;) {
        while(!pool->stop && pool->generation==generation)
            pthread_cond_wait(&pool->ready,&pool->mutex);
        if(pool->stop)break;
        generation=pool->generation;
        if(worker->index>=pool->active)continue;
        void *job=(char *)pool->jobs+worker->index*pool->stride;
        void *(*execute)(void *)=pool->execute;
        pthread_mutex_unlock(&pool->mutex);
        execute(job);
        pthread_mutex_lock(&pool->mutex);
        if(--pool->remaining==0)pthread_cond_signal(&pool->done);
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

/* Caller holds the GIL for admission/release of a complete operation, including
 * any Python repair callback between joined batches. Recursive use is refused. */
static int native_pool_acquire(NativePool *pool,int workers) {
    if(pool->process!=getpid()) {
        PyErr_SetString(PyExc_RuntimeError,"Native executor cannot be reused after fork");return -1;
    }
    if(pool->closed || pool->busy || workers>pool->count) {
        PyErr_SetString(PyExc_RuntimeError,"Native executor is closed, busy, or undersized");return -1;
    }
    pool->busy=1;return 0;
}

/* GIL detached. Stack jobs stay alive until every active worker has joined the
 * batch; the pool retains no input/output references between batches. */
static void native_pool_dispatch(NativePool *pool,void *jobs,size_t stride,
                                 int active,void *(*execute)(void *)) {
    pthread_mutex_lock(&pool->mutex);
    pool->jobs=jobs;pool->stride=stride;pool->execute=execute;
    pool->active=active;pool->remaining=active;pool->generation++;
    pthread_cond_broadcast(&pool->ready);
    while(pool->remaining)pthread_cond_wait(&pool->done,&pool->mutex);
    pool->jobs=NULL;pool->execute=NULL;pool->active=0;
    pthread_mutex_unlock(&pool->mutex);
}

static void native_pool_shutdown(NativePool *pool) {
    if(pool->closed)return;
    pool->closed=1;
    if(pool->process!=getpid())return; /* no surviving worker in the fork child */
    pthread_mutex_lock(&pool->mutex);
    pool->stop=1;pthread_cond_broadcast(&pool->ready);
    pthread_mutex_unlock(&pool->mutex);
    Py_BEGIN_ALLOW_THREADS
    for(int i=0;i<pool->started;i++)pthread_join(pool->threads[i],NULL);
    Py_END_ALLOW_THREADS
    pthread_cond_destroy(&pool->ready);pthread_cond_destroy(&pool->done);
    pthread_mutex_destroy(&pool->mutex);
}
static PyObject *native_pool_close(NativePool *pool,PyObject *unused) {
    if(pool->busy) {
        PyErr_SetString(PyExc_RuntimeError,"Cannot close a busy native executor");return NULL;
    }
    native_pool_shutdown(pool);Py_RETURN_NONE;
}
static void native_pool_dealloc(NativePool *pool) {
    native_pool_shutdown(pool);Py_TYPE(pool)->tp_free((PyObject *)pool);
}
static PyMethodDef native_pool_methods[]={
    {"close",(PyCFunction)native_pool_close,METH_NOARGS,"Join and close the instance workers"},
    {NULL,NULL,0,NULL}};
static PyTypeObject NativePoolType={
    PyVarObject_HEAD_INIT(NULL,0)
    .tp_name="agentfem._body_pipeline._NativePool",
    .tp_basicsize=sizeof(NativePool),
    .tp_dealloc=(destructor)native_pool_dealloc,
    .tp_flags=Py_TPFLAGS_DEFAULT,
    .tp_methods=native_pool_methods,
};
static PyObject *native_pool_create(PyObject *self,PyObject *argument) {
    long count=PyLong_AsLong(argument);
    if(PyErr_Occurred())return NULL;
    if(PyBool_Check(argument)||count<1||count>16) {
        PyErr_SetString(PyExc_ValueError,"Native executor requires 1..16 workers");return NULL;
    }
    NativePool *pool=PyObject_New(NativePool,&NativePoolType);
    if(!pool)return NULL;
    pool->count=(int)count;pool->started=0;pool->stop=pool->closed=pool->busy=0;
    pool->active=pool->remaining=0;pool->generation=0;pool->process=getpid();
    pool->jobs=NULL;pool->execute=NULL;
    if(pthread_mutex_init(&pool->mutex,NULL))goto mutex_failed;
    if(pthread_cond_init(&pool->ready,NULL))goto ready_failed;
    if(pthread_cond_init(&pool->done,NULL))goto done_failed;
    for(int i=0;i<count;i++) {
        pool->arguments[i]=(NativeWorker){pool,i};
        if(pthread_create(&pool->threads[i],NULL,native_pool_worker,&pool->arguments[i])) {
            Py_DECREF(pool);PyErr_SetString(PyExc_RuntimeError,"Native worker creation failed");return NULL;
        }
        pool->started++;
    }
    return (PyObject *)pool;
done_failed: pthread_cond_destroy(&pool->ready);
ready_failed: pthread_mutex_destroy(&pool->mutex);
mutex_failed: PyObject_Del(pool);
    PyErr_SetString(PyExc_RuntimeError,"Native synchronization initialization failed");return NULL;
}
