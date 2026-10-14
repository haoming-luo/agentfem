/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Generic Q1 arithmetic; uncertain predicates retain the reference fallback.
 */
#include <math.h>
#include <float.h>
#include <stdint.h>
/* acceleration of the same two-plane Q1 equations. Uncertain cases
 * request the Python reference; no triangular geometry or fast math. */
static double dot(const double *a,const double *b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static void cross(const double *a,const double *b,double *c){
 c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];}
static double det(const double *a,const double *b){return a[0]*b[1]-a[1]*b[0];}
int64_t ray_hits(int64_t n,const double *x,const double *p,const double *r,double *hits){
 double axis[3]={0,0,0},e[3],f[3];int k=0;for(int j=1;j<3;j++)if(fabs(r[j])<fabs(r[k]))k=j;
 axis[k]=1;cross(r,axis,e);double norm=sqrt(dot(e,e));for(int j=0;j<3;j++)e[j]/=norm;cross(r,e,f);
 double tol=256*DBL_EPSILON;int64_t count=0;
 for(int64_t i=0;i<n;i++){
  const double *v=x+12*i;double a[3],b[3],c[3],d[3],scale=0;
  for(int j=0;j<3;j++){double lo=v[j],hi=v[j];for(int h=1;h<4;h++){lo=fmin(lo,v[3*h+j]);hi=fmax(hi,v[3*h+j]);}scale+=(hi-lo)*(hi-lo);}
  scale=sqrt(scale);if(!(scale>0))return -1;
  for(int j=0;j<3;j++){a[j]=(v[j]-p[j])/scale;b[j]=(v[3+j]-v[j])/scale;c[j]=(v[9+j]-v[j])/scale;d[j]=(v[6+j]-v[3+j]-v[9+j]+v[j])/scale;}
  double A[2]={dot(a,e),dot(a,f)},B[2]={dot(b,e),dot(b,f)},C[2]={dot(c,e),dot(c,f)},D[2]={dot(d,e),dot(d,f)};
  double z=det(A,B),l=det(A,D)+det(C,B),q=det(C,D),mag=fmax(fabs(z),fmax(fabs(l),fabs(q)));
  if(mag<8*tol)return -1;z/=mag;l/=mag;q/=mag;
  double roots[2];int nr=0;
  if(fabs(q)<=tol){if(fabs(q)>tol/8)return -1;if(fabs(l)<=tol){if(fabs(l)>tol/8)return -1;continue;}roots[nr++]=-z/l;}
  else{if(fabs(q)<8*tol)return -1;double disc=l*l-4*q*z;if(disc < -8*tol)continue;if(disc<=8*tol)return -1;double stable=-.5*(l+copysign(sqrt(disc),l));roots[nr++]=stable/q;roots[nr++]=z/stable;}
  for(int h=0;h<nr;h++){
   double vv=roots[h];if(vv<-1e-8||vv>1+1e-8)continue;
   if(vv<=1e-8||vv>=1-1e-8)return -1;
   double denominator[2]={B[0]+D[0]*vv,B[1]+D[1]*vv};k=fabs(denominator[1])>fabs(denominator[0]);
   if(fabs(denominator[k])<=8*tol)return -1;
   double u=-(A[k]+C[k]*vv)/denominator[k];if(u<-1e-8||u>1+1e-8)continue;if(u<=1e-8||u>=1-1e-8)return -1;
   double offset[3],residual[3],ju[3],jv[3],normal[3];
   for(int j=0;j<3;j++){offset[j]=a[j]+b[j]*u+c[j]*vv+d[j]*u*vv;ju[j]=b[j]+d[j]*vv;jv[j]=c[j]+d[j]*u;}
   double t=dot(offset,r);if(t<-1e-8)continue;if(t<=1e-8)return -1;
   for(int j=0;j<3;j++)residual[j]=offset[j]-t*r[j];if(sqrt(dot(residual,residual))>1e-12)return -1;
   cross(ju,jv,normal);double crossing=dot(normal,r);if(fabs(crossing)<=8*tol)return -1;
   if(count>=2*n)return -1;double *out=hits+5*count++;out[0]=i;out[1]=u;out[2]=vv;out[3]=t*scale;out[4]=copysign(1.,crossing);
  }
 }
 return count;
}
