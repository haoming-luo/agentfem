/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Material ABI v1: batch count, old/new F, state, properties, dt, density,
 * output stress/state/work and per-point rejection status. Trusted C only.
 * State and property lengths are supplied by the provider, never fixed here.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <float.h>
#if DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error Native body arithmetic requires IEEE binary64.
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error Native body arithmetic requires strict floating point.
#endif
#include "_hex_body_native.h"
#include "_hex_affine_native.h"
#include "_hex_auxiliary.h"
#include "_native_worker_pool.h"
typedef void (*material_fn)(int,const double*,const double*,const double*,const double*,double,double,double*,double*,double*,int*);
typedef struct {
    material_fn material;
    int n, repair, has_rejected, checked, invalid; Py_ssize_t state_width; const double *in[9]; double *out[6]; int *status; double dt,rho;
} body_job;
typedef struct {
    pthread_mutex_t mutex;
    body_job base;
    int next, tile;
} body_queue;
typedef struct {
    body_queue *queue;
    int invalid;
} queued_body_job;
static body_job body_slice(const body_job *base,int start,int count) {
    body_job part=*base;
    const Py_ssize_t iw[9]={9,9,base->state_width,0,24,24,32,1,1};
    const Py_ssize_t ow[6]={9,base->state_width,1,9,24,1};
    part.n=count;part.has_rejected=0;part.invalid=0;
    part.status=base->status+start;
    for(int k=0;k<9;k++)part.in[k]=base->in[k]+iw[k]*start;
    for(int k=0;k<6;k++)part.out[k]=base->out[k]+ow[k]*start;
    return part;
}
static int checked_body_point(body_job *j,int k) {
    const Py_ssize_t widths[6]={9,j->state_width,1,9,24,1};
    for(int h=0;h<6;h++)for(Py_ssize_t i=0;i<widths[h];i++)
        if(!isfinite(j->out[h][k*widths[h]+i]))return 0;
    const double *s=j->out[0]+9*k;double scale=DBL_MIN;
    for(int i=0;i<9;i++)scale=fmax(scale,fabs(s[i]));
    double tol=1.e-12*scale;
    return fabs(s[1]-s[3])<=tol && fabs(s[2]-s[6])<=tol &&
           fabs(s[5]-s[7])<=tol && j->out[5][k]>=0.;
}
static void *execute_body(void *raw) {
    body_job *j=raw;
    if(j->repair && !j->has_rejected)return NULL;
    if(!j->repair) {
        /* Only the external provider's outputs need sentinels. Initialize in
         * each worker's slice, not a second full-body write on the GIL thread. */
        const Py_ssize_t widths[3]={9,j->state_width,1};
        for(int h=0;h<3;h++)for(Py_ssize_t k=0;k<j->n*widths[h];k++)j->out[h][k]=NAN;
    }
    if(!j->repair) j->material(j->n,j->in[0],j->in[1],j->in[2],j->in[3],j->dt,j->rho,
                j->out[0],j->out[1],j->out[2],j->status);
    for(int k=0;k<j->n;k++) {
        /* Keep successful points local to the material pass. After the exact
         * repair callback, revisit only rejected points, never the whole
         * batch. No buffer is published until both phases have joined. */
        if(!j->repair && j->status[k]) { j->has_rejected=1; continue; }
        if(j->repair && !j->status[k]) continue;
        if(!body_cell(j->in[1]+9*k,j->out[0]+9*k,j->in[4]+24*k,j->in[5]+24*k,
              j->in[6]+32*k,j->in[7][k],j->in[8][k],j->out[3]+9*k,
              j->out[4]+24*k,j->out[5]+k)) {
            /* Invalid geometry is never presented as a material repair. */
            for(int h=0;h<9;h++)j->out[3][9*k+h]=NAN;
            for(int h=0;h<24;h++)j->out[4][24*k+h]=NAN;
            j->out[5][k]=NAN;
        }
        /* Exactly the material/body constructor checks, while the cell is
         * local to its worker. Rejected material points are checked only
         * after exact repair. Publication still waits for every worker. */
        if(j->checked && !checked_body_point(j,k))j->invalid=1;
    }
    return NULL;
}
static int execute_jobs(body_job *jobs,int workers,NativePool *pool) {
    /* Called without the GIL. A single worker needs no temporary OS thread;
     * repair needs workers only for slices containing rejected points. */
    if(workers==1) { execute_body(&jobs[0]); return 0; }
    if(pool) {
        native_pool_dispatch(pool,jobs,sizeof(body_job),workers,execute_body);
        return 0;
    }
    pthread_t threads[16]; int started=0,failed=0;
    for(int i=0;i<workers;i++) {
        if(jobs[i].repair && !jobs[i].has_rejected) continue;
        if(pthread_create(&threads[started],NULL,execute_body,&jobs[i])) {failed=1;break;}
        started++;
    }
    for(int i=0;i<started;i++) pthread_join(threads[i],NULL);
    return failed;
}
static void *execute_queued_body(void *raw) {
    queued_body_job *worker=raw;body_queue *q=worker->queue;
    for(;;) {
        pthread_mutex_lock(&q->mutex);
        int start=q->next,remaining=q->base.n-start;
        int count=remaining<q->tile?remaining:q->tile;
        q->next+=count;
        pthread_mutex_unlock(&q->mutex);
        if(!count)break;
        body_job part=body_slice(&q->base,start,count);
        execute_body(&part);
        worker->invalid|=part.invalid;
    }
    return NULL;
}
static int execute_tiled(body_job *base,int workers,int tile,NativePool *pool,int *invalid) {
    body_queue queue;queue.base=*base;queue.next=0;queue.tile=tile;
    if(pthread_mutex_init(&queue.mutex,NULL))return 1;
    queued_body_job jobs[16];
    for(int i=0;i<workers;i++){jobs[i].queue=&queue;jobs[i].invalid=0;}
    int failed=0,started=0;pthread_t threads[16];
    if(workers==1)execute_queued_body(&jobs[0]);
    else if(pool)native_pool_dispatch(pool,jobs,sizeof(queued_body_job),workers,execute_queued_body);
    else {
        for(int i=0;i<workers;i++) {
            if(pthread_create(&threads[i],NULL,execute_queued_body,&jobs[i])){failed=1;break;}
            started++;
        }
        for(int i=0;i<started;i++)pthread_join(threads[i],NULL);
    }
    for(int i=0;i<workers;i++)*invalid|=jobs[i].invalid;
    pthread_mutex_destroy(&queue.mutex);
    return failed;
}
static PyObject *run_impl(PyObject *self,PyObject *args,int checked) {
    unsigned long long ma; Py_ssize_t state_width,property_count; PyObject *inputs,*callback;
    Py_buffer buffers[9]={{0}}; PyObject *out[6]={NULL}; PyObject *result=NULL;
    int workers,*status=NULL,acquired=0,tile=0; double dt,rho;
    PyObject *executor=Py_None;NativePool *pool=NULL;
    if(!PyArg_ParseTuple(args,"KOddinnO|Oi",&ma,&inputs,&dt,&rho,&workers,&state_width,&property_count,&callback,&executor,&tile)) return NULL;
    if(!PyCallable_Check(callback)||!ma||state_width<1||property_count<0||!PyTuple_Check(inputs)||PyTuple_GET_SIZE(inputs)!=9||workers<1||workers>16||tile<0||!isfinite(dt)||dt<=0||!isfinite(rho)||rho<=0) {
        PyErr_SetString(PyExc_ValueError,"Invalid native material/body configuration"); return NULL;
    }
    if(executor!=Py_None) {
        if(Py_TYPE(executor)!=&NativePoolType) {
            PyErr_SetString(PyExc_TypeError,"Expected an instance native executor");return NULL;
        }
        pool=(NativePool *)executor;
        if(native_pool_acquire(pool,workers))return NULL;
        acquired=1;
    }
    const Py_ssize_t iw[9]={9,9,state_width,0,24,24,32,1,1},ow[6]={9,state_width,1,9,24,1};
    for(int i=0;i<9;i++) {
        if(PyObject_GetBuffer(PyTuple_GET_ITEM(inputs,i),&buffers[i],PyBUF_C_CONTIGUOUS|PyBUF_FORMAT)) goto cleanup;
        if(buffers[i].itemsize!=8||!buffers[i].format||strcmp(buffers[i].format,"d")||(uintptr_t)buffers[i].buf%8) {
            PyErr_SetString(PyExc_ValueError,"Native double body buffers required"); goto cleanup;
        }
    }
    Py_ssize_t n=buffers[0].len/72;
    if(n<1||n>INT_MAX||n>PY_SSIZE_T_MAX/256||state_width>PY_SSIZE_T_MAX/8/n||property_count>PY_SSIZE_T_MAX/8) { PyErr_SetString(PyExc_ValueError,"Invalid body count"); goto cleanup; }
    for(int i=0;i<9;i++) if(buffers[i].len!=(i==3 ? property_count : n*iw[i])*8) {
        PyErr_SetString(PyExc_ValueError,"Incompatible fused body lengths"); goto cleanup;
    }
    for(int i=0;i<6;i++) {
        out[i]=PyBytes_FromStringAndSize(NULL,n*ow[i]*8);
        if(!out[i]) goto cleanup;
    }
    status=PyMem_Calloc((size_t)n,sizeof(int));
    if(!status) { PyErr_NoMemory(); goto cleanup; }
    if(workers>n) workers=(int)n;
    body_job jobs[16]; int failed=0,tiled_invalid=0;
    for(int i=0;i<workers;i++) {
        Py_ssize_t start=n*i/workers,end=n*(i+1)/workers;
        jobs[i].repair=0; jobs[i].has_rejected=0; jobs[i].checked=checked; jobs[i].invalid=0;
        jobs[i].state_width=state_width; jobs[i].material=(material_fn)(uintptr_t)ma;
        jobs[i].n=(int)(end-start); jobs[i].dt=dt; jobs[i].rho=rho;
        jobs[i].status=status+start;
        for(int k=0;k<9;k++) jobs[i].in[k]=(const double*)buffers[k].buf+iw[k]*start;
        for(int k=0;k<6;k++) jobs[i].out[k]=(double*)PyBytes_AS_STRING(out[k])+ow[k]*start;
    }
    body_job base=jobs[0];base.n=(int)n;
    Py_BEGIN_ALLOW_THREADS
    failed=tile?execute_tiled(&base,workers,tile,pool,&tiled_invalid):execute_jobs(jobs,workers,pool);
    Py_END_ALLOW_THREADS
    if(PyErr_CheckSignals()) goto cleanup;
    if(failed) { PyErr_SetString(PyExc_RuntimeError,"Native body thread creation failed"); goto cleanup; }
    /* Join every worker before exact fallback. Output bytes remain unpublished. */
    Py_ssize_t bad=0;
    for(Py_ssize_t i=0;i<n;i++) if(status[i]) bad++;
    if(bad) {
        PyObject *indices=PyBytes_FromStringAndSize(NULL,bad*sizeof(int64_t));
        if(!indices) goto cleanup;
        int64_t *ids=(int64_t*)PyBytes_AS_STRING(indices);
        Py_ssize_t k=0;
        for(Py_ssize_t i=0;i<n;i++) if(status[i]) ids[k++]=i;
        PyObject *replacement=PyObject_CallOneArg(callback,indices);
        if(!replacement) { Py_DECREF(indices); goto cleanup; }
        Py_buffer fixed[3]={{0}}; int valid=1;
        if(!PyTuple_Check(replacement)||PyTuple_GET_SIZE(replacement)!=3) {
            PyErr_SetString(PyExc_ValueError,"Fallback must return three buffers"); valid=0;
        }
        for(int j=0;j<3 && valid;j++) {
            if(PyObject_GetBuffer(PyTuple_GET_ITEM(replacement,j),&fixed[j],PyBUF_C_CONTIGUOUS|PyBUF_FORMAT)) {valid=0;break;}
            if(fixed[j].itemsize!=8||!fixed[j].format||strcmp(fixed[j].format,"d")||(uintptr_t)fixed[j].buf%8||fixed[j].len!=bad*ow[j]*8) {
                PyErr_SetString(PyExc_ValueError,"Invalid fallback buffer");valid=0;break;
            }
            const double *v=fixed[j].buf;
            for(Py_ssize_t i=0;i<bad*ow[j];i++) if(!isfinite(v[i])) {
                PyErr_SetString(PyExc_ValueError,"Nonfinite fallback output");valid=0;break;
            }
        }
        if(valid) for(int j=0;j<3;j++) for(Py_ssize_t i=0;i<bad;i++)
            memcpy(PyBytes_AS_STRING(out[j])+ids[i]*ow[j]*8,(const char*)fixed[j].buf+i*ow[j]*8,ow[j]*8);
        for(int j=0;j<3;j++) if(fixed[j].obj) PyBuffer_Release(&fixed[j]);
        Py_DECREF(replacement);Py_DECREF(indices);
        if(!valid) goto cleanup;
        for(int j=0;j<workers;j++) {
            jobs[j].repair=1;
            /* Dynamic tiles have no persistent worker-to-slice assignment.
             * Rebuild repair ownership from the joined per-point status. */
            if(tile)for(int k=0;k<jobs[j].n;k++)
                if(jobs[j].status[k]){jobs[j].has_rejected=1;break;}
        }
        Py_BEGIN_ALLOW_THREADS
        failed=execute_jobs(jobs,workers,pool);
        Py_END_ALLOW_THREADS
        if(PyErr_CheckSignals()) goto cleanup;
        if(failed) {PyErr_SetString(PyExc_RuntimeError,"Native repair thread creation failed");goto cleanup;}
    }
    for(int j=0;j<workers;j++)if(jobs[j].invalid||tiled_invalid) {
        PyErr_SetString(PyExc_ValueError,"Nonfinite or inadmissible native material/body output");goto cleanup;
    }
    /* The Python material/body constructors validate all immutable buffers
     * before they can enter a trial. Avoid an identical full-array scan here. */
    result=PyTuple_New(6);
    if(result) for(int i=0;i<6;i++) PyTuple_SET_ITEM(result,i,Py_NewRef(out[i]));
cleanup:
    if(acquired)pool->busy=0;
    PyMem_Free(status);
    for(int i=0;i<6;i++) Py_XDECREF(out[i]);
    for(int i=0;i<9;i++) if(buffers[i].obj) PyBuffer_Release(&buffers[i]);
    return result;
}
static PyObject *run(PyObject *self,PyObject *args) {return run_impl(self,args,0);}
static PyObject *run_checked(PyObject *self,PyObject *args) {return run_impl(self,args,1);}
#include "_hex_auxiliary_dispatch.h"
static PyMethodDef methods[]={
    {"create_pool",native_pool_create,METH_O,"Create an instance-owned sleeping native executor"},
    {"run",run,METH_VARARGS,"Trusted C material ABI with core-owned immutable body output"},
    {"run_checked",run_checked,METH_VARARGS,"Worker-validated immutable material and body output"},
    {"run_tiled_checked",run_checked,METH_VARARGS,"Opt-in dynamically tiled, worker-validated body execution"},
    {"auxiliary",auxiliary,METH_VARARGS,"Joined disjoint-cell arithmetic with immutable output"},
    {NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_body_pipeline",NULL,-1,methods};
PyMODINIT_FUNC PyInit__body_pipeline(void) {
    if(PyType_Ready(&NativePoolType)<0)return NULL;
    return PyModule_Create(&module);
}
