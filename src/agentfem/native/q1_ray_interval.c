/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Generic Q1 arithmetic; uncertain predicates retain the reference fallback.
 */
#include <math.h>
#include <stdint.h>
/* interval filter: every arithmetic operation rounds outwards.
 * Only globally monotone elimination polynomials are admitted. */
typedef struct {double lo,hi;} I;
static I value(double x){return (I){x,x};}
static I add(I a,I b){return (I){nextafter(a.lo+b.lo,-INFINITY),nextafter(a.hi+b.hi,INFINITY)};}
static I neg(I a){return (I){-a.hi,-a.lo};}
static I sub(I a,I b){return add(a,neg(b));}
static I mul(I a,I b){double p[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};for(int i=0;i<4;i++)if(!isfinite(p[i]))return (I){-INFINITY,INFINITY};double l=p[0],h=p[0];for(int i=1;i<4;i++){l=fmin(l,p[i]);h=fmax(h,p[i]);}return (I){nextafter(l,-INFINITY),nextafter(h,INFINITY)};}
static int valid(I a){return isfinite(a.lo)&&isfinite(a.hi)&&a.lo<=a.hi;}
static int sign(I a){return a.lo>0?1:a.hi<0?-1:0;}
static I divide(I a,I b){if(!sign(b))return (I){-INFINITY,INFINITY};return mul(a,(I){nextafter(1/b.hi,-INFINITY),nextafter(1/b.lo,INFINITY)});}
static I dot(I*a,I*b){I s=mul(a[0],b[0]);for(int i=1;i<3;i++)s=add(s,mul(a[i],b[i]));return s;}
static I det(I*a,I*b){return sub(mul(a[0],b[1]),mul(a[1],b[0]));}
static I polynomial(I z,I l,I q,I v){return add(z,mul(v,add(l,mul(q,v))));}

int64_t interval_ray_hits(int64_t n,const double*x,const double*p,const double*r,double*out){
 int k=0;for(int i=1;i<3;i++)if(fabs(r[i])>fabs(r[k]))k=i;
 int j0=(k+1)%3,j1=(k+2)%3;I ray[3]={value(r[0]),value(r[1]),value(r[2])};
 I rr=dot(ray,ray);int64_t count=0;
 for(int64_t f=0;f<n;f++){
  I a[3],b[3],c[3],d[3],A[2],B[2],C[2],D[2];const double*v=x+12*f;
  for(int j=0;j<3;j++){
   a[j]=sub(value(v[j]),value(p[j]));b[j]=sub(value(v[3+j]),value(v[j]));
   c[j]=sub(value(v[9+j]),value(v[j]));d[j]=sub(sub(value(v[6+j]),value(v[9+j])),b[j]);
  }
  for(int h=0;h<2;h++){
   int j=h?j1:j0;
   A[h]=sub(mul(ray[k],a[j]),mul(ray[j],a[k]));
   B[h]=sub(mul(ray[k],b[j]),mul(ray[j],b[k]));
   C[h]=sub(mul(ray[k],c[j]),mul(ray[j],c[k]));
   D[h]=sub(mul(ray[k],d[j]),mul(ray[j],d[k]));
  }
  I z=det(A,B),l=add(det(A,D),det(C,B)),q=det(C,D);
  I derivative=add(l,mul(value(2),mul(q,(I){0,1})));
  if(!valid(derivative)||!sign(derivative))return -1;
  I f0=z,f1=polynomial(z,l,q,value(1));
  if(!valid(f0)||!valid(f1)||!sign(f0)||!sign(f1))return -1;
  if(sign(f0)==sign(f1))continue;
  I vv={0,1};
  for(int iteration=0;iteration<5;iteration++){
   double m=.5*(vv.lo+vv.hi);I df=add(l,mul(value(2),mul(q,vv)));
   I image=sub(value(m),divide(polynomial(z,l,q,value(m)),df));
   vv.lo=fmax(vv.lo,image.lo);vv.hi=fmin(vv.hi,image.hi);
   if(!valid(vv))return -1;
  }
  if(!(vv.lo>0&&vv.hi<1))return -1;
  I denom[2]={add(B[0],mul(D[0],vv)),add(B[1],mul(D[1],vv))};
  int h=sign(denom[0])?0:1;if(!sign(denom[h]))return -1;
  I u=divide(neg(add(A[h],mul(C[h],vv))),denom[h]);
  if(!valid(u))return -1;
  if(u.hi<0||u.lo>1)continue;
  if(!(u.lo>0&&u.hi<1))return -1;
  I delta[3],du[3],dv[3],cross[3];
  for(int j=0;j<3;j++){
   delta[j]=add(add(a[j],mul(b[j],u)),add(mul(c[j],vv),mul(d[j],mul(u,vv))));
   du[j]=add(b[j],mul(d[j],vv));dv[j]=add(c[j],mul(d[j],u));
  }
  I t=divide(dot(delta,ray),rr);if(!valid(t))return -1;
  if(t.hi<0)continue;if(t.lo<=0)return -1;
  for(int j=0;j<3;j++)cross[j]=sub(mul(du[(j+1)%3],dv[(j+2)%3]),mul(du[(j+2)%3],dv[(j+1)%3]));
  I orientation=dot(cross,ray);if(!valid(orientation)||!sign(orientation))return -1;
  double *record=out+5*count++;record[0]=f;record[1]=.5*(u.lo+u.hi);record[2]=.5*(vv.lo+vv.hi);record[3]=.5*(t.lo+t.hi);record[4]=sign(orientation);
 }
 return count;
}
