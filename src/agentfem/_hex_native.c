/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Sufficient geometry/orientation screens and ordered nodal force reduction.
 * Uncertified geometry retains the Python/LU/Bernstein reference fallback.
 * No fast-math: finite admission and conservative roundoff guards are required.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#if DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error Native kernels require IEEE-754 binary64; use the Python fallback.
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error Native admission requires strict finite arithmetic, not fast-math.
#endif

static PyObject *screen(PyObject *self, PyObject *args) {
    PyObject *objects[5]; Py_buffer b[5] = {{0}};
    if (!PyArg_ParseTuple(args,"OOOOO",&objects[0],&objects[1],&objects[2],&objects[3],&objects[4])) return NULL;
    int acquired=0;
    for (; acquired<5; ++acquired) {
        if (PyObject_GetBuffer(objects[acquired],&b[acquired],PyBUF_CONTIG_RO|PyBUF_FORMAT)<0) goto fail;
    }
    for (int i=0;i<5;++i) {
        Py_ssize_t width=(i==4 ? sizeof(int):sizeof(double));
        if (b[i].itemsize!=width || !b[i].format || strcmp(b[i].format,i==4 ? "i":"d") ||
            (uintptr_t)b[i].buf%width) {
            PyErr_SetString(PyExc_ValueError,"Affine screen requires aligned native float64/int32 buffers");goto fail;
        }
    }
    Py_ssize_t n=b[3].len/(Py_ssize_t)sizeof(double);
    if (b[3].len%sizeof(double) || b[0].len!=n*24*sizeof(double) ||
        b[1].len!=b[0].len || b[2].len!=n*9*sizeof(double) || b[4].len!=24*sizeof(int)) {
        PyErr_SetString(PyExc_ValueError,"Invalid affine screen buffer lengths"); goto fail;
    }
    PyObject *result=PyBytes_FromStringAndSize(NULL,n);
    if (!result) goto fail;
    const double *xx=b[0].buf,*uu=b[1].buf,*ff=b[2].buf,*radius=b[3].buf;
    const int *edges=b[4].buf;
    for (int j=0;j<24;++j) if (edges[j]<0 || edges[j]>=8) {
        Py_DECREF(result); PyErr_SetString(PyExc_ValueError,"Invalid affine edge index");goto fail;
    }
    char *out=PyBytes_AS_STRING(result);
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t k=0;k<n;++k) {
        const double *x=xx+24*k,*u=uu+24*k,*f=ff+9*k;
        double d[9],g[9],rem[24],orient=0.,rho=0.,gnorm=0.,xs=0.,us=0.;
        int valid=isfinite(radius[k]) && radius[k]>0.; out[k]=0;
        for (int i=0;i<9;++i) {d[i]=f[i]-(i%4==0 ? 1.:0.);valid &= isfinite(f[i]);}
        for (int i=0;i<24;++i) {
            valid &= isfinite(x[i]) && isfinite(u[i]);
            xs=fmax(xs,fabs(x[i]));us=fmax(us,fabs(u[i]));
        }
        if (!valid) continue;
        for(int i=0;i<3;++i) {
            double row=0.;for(int j=0;j<3;++j)row+=fabs(d[3*i+j]);orient=fmax(orient,row);
        }
        /* Positive orientation by a strict near-identity homotopy, not Gram alone. */
        if (!isfinite(orient) || orient>=.5) continue;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
            double value=0.;for(int l=0;l<3;++l)value+=f[3*l+i]*f[3*l+j];g[3*i+j]=value;
        }
        for(int i=0;i<3;++i) {
            double row=0.,delta=0.;for(int j=0;j<3;++j) {
                row+=fabs(g[3*i+j]);delta+=fabs(g[3*i+j]-(i==j ? 1.:0.));
            }
            gnorm=fmax(gnorm,row);rho=fmax(rho,delta);
        }
        rho+=256.*DBL_EPSILON*(1.+gnorm);
        if (!isfinite(rho) || rho>=.5) continue;
        for(int a=0;a<8;++a)for(int i=0;i<3;++i) {
            double value=0.;for(int j=0;j<3;++j)value+=d[3*i+j]*(x[3*a+j]-x[j]);
            rem[3*a+i]=(u[3*a+i]-u[i])-value;
        }
        double bound=0.;
        for(int axis=0;axis<3;++axis) {
            double maximum=0.;
            for(int edge=0;edge<4;++edge) {
                int a=edges[8*axis+2*edge],b=edges[8*axis+2*edge+1];double squared=0.;
                for(int i=0;i<3;++i) {double delta=rem[3*b+i]-rem[3*a+i];squared+=delta*delta;}
                if (!isfinite(squared)) valid=0;
                maximum=fmax(maximum,squared);
            }
            bound+=maximum;
        }
        double guard=512.*DBL_EPSILON*(xs*(1.+sqrt(1.+rho))+us);
        double total=sqrt(bound)+guard;
        out[k]=(valid && isfinite(total) && total<radius[k]*sqrt(1.-rho)) ? 1:0;
    }
    Py_END_ALLOW_THREADS
    for(int i=0;i<acquired;++i)PyBuffer_Release(&b[i]);
    return result;
fail:
    for(int i=0;i<acquired;++i)PyBuffer_Release(&b[i]);
    return NULL;
}
static PyObject *orientation(PyObject *self, PyObject *object) {
    Py_buffer b={0};
    if(PyObject_GetBuffer(object,&b,PyBUF_CONTIG_RO|PyBUF_FORMAT)<0)return NULL;
    if(b.itemsize!=sizeof(double) || !b.format || strcmp(b.format,"d") ||
       b.len%(9*sizeof(double)) || (uintptr_t)b.buf%sizeof(double)) {
        PyBuffer_Release(&b);PyErr_SetString(PyExc_ValueError,"Orientation requires aligned native float64 matrices");return NULL;
    }
    Py_ssize_t n=b.len/(9*sizeof(double));
    PyObject *result=PyBytes_FromStringAndSize(NULL,n);
    if(!result){PyBuffer_Release(&b);return NULL;}
    const double *f=b.buf;char *out=PyBytes_AS_STRING(result);
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t k=0;k<n;++k) {
        int finite=1;double radius=0.;
        for(int i=0;i<9;++i)finite &= isfinite(f[9*k+i]);
        for(int i=0;i<3;++i) {
            const double *row=f+9*k+3*i;
            double value=(fabs(row[0]-(i==0))+fabs(row[1]-(i==1)))+fabs(row[2]-(i==2));
            radius=fmax(radius,value);
        }
        out[k]=!finite ? 2 : (radius<.5 ? 1:0);
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&b);return result;
}

static PyObject *scatter(PyObject *self, PyObject *args) {
    PyObject *force,*nodes,*extra;Py_ssize_t count;Py_buffer f={0},n={0},e={0};
    if(!PyArg_ParseTuple(args,"OOOn",&force,&nodes,&extra,&count))return NULL;
    if(PyObject_GetBuffer(force,&f,PyBUF_CONTIG_RO|PyBUF_FORMAT)<0)return NULL;
    if(PyObject_GetBuffer(nodes,&n,PyBUF_CONTIG_RO|PyBUF_FORMAT)<0){PyBuffer_Release(&f);return NULL;}
    if(extra!=Py_None && PyObject_GetBuffer(extra,&e,PyBUF_CONTIG_RO|PyBUF_FORMAT)<0)goto error;
    if(count<0 || count>PY_SSIZE_T_MAX/(3*sizeof(double)) || f.itemsize!=sizeof(double) ||
       !f.format || strcmp(f.format,"d") || (n.itemsize!=4 && n.itemsize!=8) ||
       !n.format || (strcmp(n.format,"i") && strcmp(n.format,"l") && strcmp(n.format,"q")) ||
       n.len%n.itemsize || f.len!=n.len/n.itemsize*3*sizeof(double) ||
       (uintptr_t)f.buf%sizeof(double) || (uintptr_t)n.buf%n.itemsize ||
       (extra!=Py_None && (e.len!=f.len || e.itemsize!=sizeof(double) || !e.format ||
         strcmp(e.format,"d") || (uintptr_t)e.buf%sizeof(double)))) {
        PyErr_SetString(PyExc_ValueError,"Scatter requires aligned float64 forces and signed node indices");goto error;
    }
    PyObject *result=PyBytes_FromStringAndSize(NULL,count*3*sizeof(double));
    if(!result)goto error;
    double *out=(double *)PyBytes_AS_STRING(result);
    memset(out,0,count*3*sizeof(double));
    const double *forces=f.buf,*additional=e.buf;
    Py_ssize_t points=n.len/n.itemsize;int valid=1;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t p=0;p<points;++p) {
        int64_t id=n.itemsize==4 ? ((const int32_t *)n.buf)[p]:((const int64_t *)n.buf)[p];
        if(id<0 || id>=count){valid=0;break;}
        for(int i=0;i<3;++i) {
            double a=forces[3*p+i],b=additional ? additional[3*p+i]:0.;
            if(!isfinite(a)||!isfinite(b)){valid=0;break;}
            /* Match bincount's node-wise input order and addition of contributions. */
            out[3*id+i]+=additional ? a+b:a;
        }
        if(!valid)break;
    }
    for(Py_ssize_t p=0;valid && p<3*count;++p)if(!isfinite(out[p]))valid=0;
    Py_END_ALLOW_THREADS
    if(!valid){Py_DECREF(result);PyErr_SetString(PyExc_ValueError,"Invalid index or nonfinite scatter force");goto error;}
    PyBuffer_Release(&f);PyBuffer_Release(&n);if(e.obj)PyBuffer_Release(&e);return result;
error:
    PyBuffer_Release(&f);PyBuffer_Release(&n);if(e.obj)PyBuffer_Release(&e);return NULL;
}

static PyObject *all_finite(PyObject *self, PyObject *object) {
    Py_buffer b={0};
    if(PyObject_GetBuffer(object,&b,PyBUF_CONTIG_RO|PyBUF_FORMAT)<0)return NULL;
    if(b.itemsize!=sizeof(double) || !b.format ||
       (strcmp(b.format,"d") && strcmp(b.format,"=d")) || b.len%sizeof(double)) {
        PyBuffer_Release(&b);PyErr_SetString(PyExc_ValueError,"Finite scan requires native float64 storage");return NULL;
    }
    const unsigned char *data=b.buf;Py_ssize_t n=b.len/sizeof(double);unsigned int invalid=0;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t i=0;i<n;++i) {
        uint64_t bits;memcpy(&bits,data+8*i,8);
        invalid |= (bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000);
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&b);return PyBool_FromLong(!invalid);
}

static PyObject *cell_work(PyObject *self, PyObject *args) {
    PyObject *force,*nodes,*increment;Py_ssize_t width;Py_buffer f={0},n={0},u={0};
    if(!PyArg_ParseTuple(args,"OOOn",&force,&nodes,&increment,&width))return NULL;
    if(PyObject_GetBuffer(force,&f,PyBUF_CONTIG_RO|PyBUF_FORMAT)<0)return NULL;
    if(PyObject_GetBuffer(nodes,&n,PyBUF_CONTIG_RO|PyBUF_FORMAT)<0){PyBuffer_Release(&f);return NULL;}
    if(PyObject_GetBuffer(increment,&u,PyBUF_CONTIG_RO|PyBUF_FORMAT)<0)goto work_error;
    if(width<1 || width>1024 || f.itemsize!=8 || !f.format || strcmp(f.format,"d") ||
       u.itemsize!=8 || !u.format || strcmp(u.format,"d") || u.len%24 ||
       (n.itemsize!=4 && n.itemsize!=8) || !n.format ||
       (strcmp(n.format,"i")&&strcmp(n.format,"l")&&strcmp(n.format,"q")) ||
       n.len%(8*n.itemsize) || f.len!=n.len/n.itemsize*24 ||
       (uintptr_t)f.buf%8 || (uintptr_t)u.buf%8 || (uintptr_t)n.buf%n.itemsize) {
        PyErr_SetString(PyExc_ValueError,"Invalid cell work storage");goto work_error;
    }
    Py_ssize_t cells=n.len/(8*n.itemsize),nn=u.len/24,blocks=(cells+width-1)/width;
    PyObject *signed_result=PyBytes_FromStringAndSize(NULL,blocks*8);
    PyObject *absolute_result=PyBytes_FromStringAndSize(NULL,blocks*8);
    if(!signed_result || !absolute_result){Py_XDECREF(signed_result);Py_XDECREF(absolute_result);goto work_error;}
    double *signed_out=(double *)PyBytes_AS_STRING(signed_result),*absolute_out=(double *)PyBytes_AS_STRING(absolute_result);
    const double *forces=f.buf,*motion=u.buf;int valid=1;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t begin=0,block=0;begin<cells && valid;begin+=width,++block) {
        double sums[1024],absolute[1024];Py_ssize_t count=cells-begin;if(count>width)count=width;
        for(Py_ssize_t i=0;i<count && valid;++i) {
            double products[24],magnitudes[24];
            for(int a=0;a<8 && valid;++a) {
                Py_ssize_t point=8*(begin+i)+a;
                int64_t node=n.itemsize==4 ? ((const int32_t *)n.buf)[point]:((const int64_t *)n.buf)[point];
                if(node<0 || node>=nn){valid=0;break;}
                for(int j=0;j<3;++j) {
                    double value=forces[3*point+j]*motion[3*node+j];
                    if(!isfinite(value)){valid=0;break;}
                    products[3*a+j]=value;magnitudes[3*a+j]=fabs(value);
                }
            }
            if(!valid)break;
            for(int count2=24;count2>1;count2=(count2+1)/2)for(int a=0;a<(count2+1)/2;++a) {
                products[a]=products[2*a]+(2*a+1<count2 ? products[2*a+1]:0.);
                magnitudes[a]=magnitudes[2*a]+(2*a+1<count2 ? magnitudes[2*a+1]:0.);
            }
            sums[i]=products[0];absolute[i]=magnitudes[0];
        }
        if(!valid)break;
        for(Py_ssize_t count2=count;count2>1;count2=(count2+1)/2)for(Py_ssize_t a=0;a<(count2+1)/2;++a) {
            sums[a]=sums[2*a]+(2*a+1<count2 ? sums[2*a+1]:0.);
            absolute[a]=absolute[2*a]+(2*a+1<count2 ? absolute[2*a+1]:0.);
        }
        signed_out[block]=sums[0];absolute_out[block]=absolute[0];
        if(!isfinite(sums[0]) || !isfinite(absolute[0]))valid=0;
    }
    Py_END_ALLOW_THREADS
    if(!valid){Py_DECREF(signed_result);Py_DECREF(absolute_result);PyErr_SetString(PyExc_ValueError,"Invalid index or nonfinite cell work");goto work_error;}
    PyBuffer_Release(&f);PyBuffer_Release(&n);PyBuffer_Release(&u);
    return Py_BuildValue("NN",signed_result,absolute_result);
work_error:
    PyBuffer_Release(&f);PyBuffer_Release(&n);if(u.obj)PyBuffer_Release(&u);return NULL;
}

static PyObject *determinants(PyObject *self, PyObject *object) {
    Py_buffer b={0};
    if(PyObject_GetBuffer(object,&b,PyBUF_CONTIG_RO|PyBUF_FORMAT)<0)return NULL;
    if(b.itemsize!=8 || !b.format || strcmp(b.format,"d") || b.len%72 || (uintptr_t)b.buf%8) {
        PyBuffer_Release(&b);PyErr_SetString(PyExc_ValueError,"Determinants require aligned native float64 matrices");return NULL;
    }
    Py_ssize_t count=b.len/72;
    PyObject *values=PyBytes_FromStringAndSize(NULL,count*8),*states=PyBytes_FromStringAndSize(NULL,count);
    if(!values || !states){Py_XDECREF(values);Py_XDECREF(states);PyBuffer_Release(&b);return NULL;}
    double *out=(double *)PyBytes_AS_STRING(values);char *status=PyBytes_AS_STRING(states);const double *f=b.buf;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t i=0;i<count;++i) {
        const double *a=f+9*i;int finite=1,tiny=0;
        for(int k=0;k<9;++k){finite &= isfinite(a[k]);tiny |= a[k]!=0. && fabs(a[k])<sqrt(DBL_MIN);}
        if(!finite){status[i]=2;out[i]=NAN;continue;}
        double t[6]={a[0]*a[4]*a[8],a[1]*a[5]*a[6],a[2]*a[3]*a[7],
                     a[2]*a[4]*a[6],a[1]*a[3]*a[8],a[0]*a[5]*a[7]};
        out[i]=t[0]+t[1]+t[2]-t[3]-t[4]-t[5];
        double absolute=0.;for(int k=0;k<6;++k)absolute+=fabs(t[k]);
        status[i]=(!isfinite(absolute)||!isfinite(out[i])||tiny||fabs(out[i])<=.01*absolute||absolute<DBL_MIN*128) ? 0:1;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&b);return Py_BuildValue("NN",values,states);
}

#include "_hex_body_native.h"

static PyMethodDef methods[]={
    {"body_response",body_response,METH_VARARGS,"Material-independent Piola, Hex8 force and hourglass contraction."},
    {"affine_screen",screen,METH_VARARGS,"Conservative cell-local affine certificate mask."},
    {"orientation_screen",orientation,METH_O,"Near-identity orientation: 1 admitted, 0 use LU, 2 nonfinite."},
    {"scatter",scatter,METH_VARARGS,"Ordered finite nodal force reduction."},
    {"all_finite",all_finite,METH_O,"Exact binary64 finite scan without temporary masks."},
    {"cell_work",cell_work,METH_VARARGS,"Pairwise force-displacement work blocks with checked indices."},
    {"determinants",determinants,METH_O,"Six-product determinants with finite/cancellation/LU flags."},
    {NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_hex_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__hex_native(void){return PyModule_Create(&module);}
