/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Material-independent Hex8 arithmetic. No state layout or material constants.
 * Included by _hex_native.c; all storage remains detached until publication.
 */
static void body_cofactor(const double *f, double *c) {
    c[0]=f[4]*f[8]-f[7]*f[5]; c[1]=f[5]*f[6]-f[8]*f[3]; c[2]=f[3]*f[7]-f[6]*f[4];
    c[3]=f[7]*f[2]-f[1]*f[8]; c[4]=f[8]*f[0]-f[2]*f[6]; c[5]=f[6]*f[1]-f[0]*f[7];
    c[6]=f[1]*f[5]-f[4]*f[2]; c[7]=f[2]*f[3]-f[5]*f[0]; c[8]=f[0]*f[4]-f[3]*f[1];
}

static int body_cell(const double *f,const double *s,const double *u,const double *g,
 const double *gamma,double volume,double coefficient,double *p,double *force,double *energy_out) {
        double c[9],modes[12]={0};body_cofactor(f,c);
        double determinant=(f[0]*c[0]+f[3]*c[3])+f[6]*c[6];
        if(!(determinant>0) || !isfinite(determinant) || !(volume>0) || coefficient<0) return 0;
        for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
            double v=0;for(int h=0;h<3;h++)v+=s[3*i+h]*c[3*h+j];p[3*i+j]=v;
        }
        for(int a=0;a<8;a++)for(int m=0;m<4;m++)for(int i=0;i<3;i++)
            modes[3*m+i]+=gamma[4*a+m]*u[3*a+i];
        double energy=0;for(int m=0;m<4;m++)for(int i=0;i<3;i++)energy+=modes[3*m+i]*modes[3*m+i];
        *energy_out=.5*coefficient*energy;
        for(int a=0;a<8;a++)for(int i=0;i<3;i++) {
            double v=0,hg=0;
            for(int j=0;j<3;j++)v+=g[3*a+j]*p[3*i+j];
            for(int m=0;m<4;m++)hg+=gamma[4*a+m]*modes[3*m+i];
            force[3*a+i]=volume*v+coefficient*hg;
        }
    return 1;
}

static PyObject *body_response(PyObject *self, PyObject *args) {
    Py_buffer b[7]={{0}}; PyObject *out[3]={NULL},*result=NULL;
    const Py_ssize_t widths[7]={9,9,24,24,32,1,1}, outputs[3]={9,24,1};
    if(!PyTuple_Check(args) || PyTuple_GET_SIZE(args)!=7) {
        PyErr_SetString(PyExc_TypeError,"body_response needs F, stress, u, gradient, modes, volume, coefficient");return NULL;
    }
    Py_ssize_t n=0;
    for(int i=0;i<7;i++) {
        if(PyObject_GetBuffer(PyTuple_GET_ITEM(args,i),b+i,PyBUF_CONTIG_RO|PyBUF_FORMAT))goto done;
        if(b[i].itemsize!=8 || !b[i].format || strcmp(b[i].format,"d") || (uintptr_t)b[i].buf%8) {
            PyErr_SetString(PyExc_ValueError,"Body kernel requires aligned native float64 arrays");goto done;
        }
        if(i==0) {
            if(b[0].len%72) {PyErr_SetString(PyExc_ValueError,"Invalid gradient buffer");goto done;}
            n=b[0].len/72;
            if(n>PY_SSIZE_T_MAX/(32*8)) {PyErr_SetString(PyExc_OverflowError,"Body batch too large");goto done;}
        }
        if(b[i].len!=n*widths[i]*8) {PyErr_SetString(PyExc_ValueError,"Incompatible body buffer lengths");goto done;}
    }
    for(int i=0;i<3;i++) {out[i]=PyBytes_FromStringAndSize(NULL,n*outputs[i]*8);if(!out[i])goto done;}
    int valid=1;
    Py_BEGIN_ALLOW_THREADS
    for(int j=0;j<7 && valid;j++) {
        const double *v=b[j].buf;
        for(Py_ssize_t i=0;i<n*widths[j];i++)if(!isfinite(v[i])){valid=0;break;}
    }
    for(Py_ssize_t k=0;k<n && valid;k++) {
        const double *f=(const double*)b[0].buf+9*k,*s=(const double*)b[1].buf+9*k;
        const double *u=(const double*)b[2].buf+24*k,*g=(const double*)b[3].buf+24*k;
        const double *gamma=(const double*)b[4].buf+32*k;
        double volume=((const double*)b[5].buf)[k],coefficient=((const double*)b[6].buf)[k];
        double *p=(double*)PyBytes_AS_STRING(out[0])+9*k;
        double *force=(double*)PyBytes_AS_STRING(out[1])+24*k;
        valid=body_cell(f,s,u,g,gamma,volume,coefficient,p,force,
                        (double*)PyBytes_AS_STRING(out[2])+k);
    }
    for(int j=0;j<3 && valid;j++) {
        const double *v=(double*)PyBytes_AS_STRING(out[j]);
        for(Py_ssize_t i=0;i<n*outputs[j];i++)if(!isfinite(v[i])){valid=0;break;}
    }
    Py_END_ALLOW_THREADS
    if(!valid){PyErr_SetString(PyExc_ValueError,"Invalid geometry or nonfinite body arithmetic");goto done;}
    result=PyTuple_New(3);
    if(result)for(int j=0;j<3;j++)PyTuple_SET_ITEM(result,j,Py_NewRef(out[j]));
done:
    for(int i=0;i<7;i++)if(b[i].obj)PyBuffer_Release(b+i);
    for(int i=0;i<3;i++)Py_XDECREF(out[i]);
    return result;
}
