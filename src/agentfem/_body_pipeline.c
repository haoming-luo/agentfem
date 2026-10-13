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
#include "_hex_auxiliary.h"
typedef void (*material_fn)(int,const double*,const double*,const double*,const double*,double,double,double*,double*,double*,int*);
typedef struct {
    material_fn material;
    int n, repair, has_rejected; Py_ssize_t state_width; const double *in[9]; double *out[6]; int *status; double dt,rho;
} body_job;
static void *execute_body(void *raw) {
    body_job *j=raw;
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
    }
    return NULL;
}
static int execute_jobs(body_job *jobs,int workers) {
    /* Called without the GIL. A single worker needs no temporary OS thread;
     * repair needs workers only for slices containing rejected points. */
    if(workers==1) { execute_body(&jobs[0]); return 0; }
    pthread_t threads[16]; int started=0,failed=0;
    for(int i=0;i<workers;i++) {
        if(jobs[i].repair && !jobs[i].has_rejected) continue;
        if(pthread_create(&threads[started],NULL,execute_body,&jobs[i])) {failed=1;break;}
        started++;
    }
    for(int i=0;i<started;i++) pthread_join(threads[i],NULL);
    return failed;
}
static PyObject *run(PyObject *self,PyObject *args) {
    unsigned long long ma; Py_ssize_t state_width,property_count; PyObject *inputs,*callback;
    Py_buffer buffers[9]={{0}}; PyObject *out[6]={NULL}; PyObject *result=NULL;
    int workers,*status=NULL; double dt,rho;
    if(!PyArg_ParseTuple(args,"KOddinnO",&ma,&inputs,&dt,&rho,&workers,&state_width,&property_count,&callback)) return NULL;
    if(!PyCallable_Check(callback)||!ma||state_width<1||property_count<0||!PyTuple_Check(inputs)||PyTuple_GET_SIZE(inputs)!=9||workers<1||workers>16||!isfinite(dt)||dt<=0||!isfinite(rho)||rho<=0) {
        PyErr_SetString(PyExc_ValueError,"Invalid native material/body configuration"); return NULL;
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
    body_job jobs[16]; int failed=0;
    for(int i=0;i<workers;i++) {
        Py_ssize_t start=n*i/workers,end=n*(i+1)/workers;
        jobs[i].repair=0; jobs[i].has_rejected=0; jobs[i].state_width=state_width; jobs[i].material=(material_fn)(uintptr_t)ma;
        jobs[i].n=(int)(end-start); jobs[i].dt=dt; jobs[i].rho=rho;
        jobs[i].status=status+start;
        for(int k=0;k<9;k++) jobs[i].in[k]=(const double*)buffers[k].buf+iw[k]*start;
        for(int k=0;k<6;k++) jobs[i].out[k]=(double*)PyBytes_AS_STRING(out[k])+ow[k]*start;
    }
    Py_BEGIN_ALLOW_THREADS
    failed=execute_jobs(jobs,workers);
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
        for(int j=0;j<workers;j++) jobs[j].repair=1;
        Py_BEGIN_ALLOW_THREADS
        failed=execute_jobs(jobs,workers);
        Py_END_ALLOW_THREADS
        if(PyErr_CheckSignals()) goto cleanup;
        if(failed) {PyErr_SetString(PyExc_RuntimeError,"Native repair thread creation failed");goto cleanup;}
    }
    /* The Python material/body constructors validate all immutable buffers
     * before they can enter a trial. Avoid an identical full-array scan here. */
    result=PyTuple_New(6);
    if(result) for(int i=0;i<6;i++) PyTuple_SET_ITEM(result,i,Py_NewRef(out[i]));
cleanup:
    PyMem_Free(status);
    for(int i=0;i<6;i++) Py_XDECREF(out[i]);
    for(int i=0;i<9;i++) if(buffers[i].obj) PyBuffer_Release(&buffers[i]);
    return result;
}
#include "_hex_auxiliary_dispatch.h"
static PyMethodDef methods[]={
    {"run",run,METH_VARARGS,"Trusted C material ABI with core-owned immutable body output"},
    {"auxiliary",auxiliary,METH_VARARGS,"Joined disjoint-cell arithmetic with immutable output"},
    {NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_body_pipeline",NULL,-1,methods};
PyMODINIT_FUNC PyInit__body_pipeline(void) { return PyModule_Create(&module); }
