/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Generic Q1 arithmetic; uncertain predicates retain the reference fallback.
 */
#include <math.h>
#include <float.h>
#if !defined(Q1_ADAPTIVE_MARGIN) || !Q1_ADAPTIVE_MARGIN
#error Packaged Q1 geometry requires the scale-conditioned margin policy.
#endif

static double dot(const double *a,const double *b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static void delta(const double *a,const double *b,const double *c,const double *d,
                  double u,double v,double *z,double *ju,double *jv){
 for(int i=0;i<3;i++){z[i]=a[i]+b[i]*u+c[i]*v+d[i]*u*v;ju[i]=b[i]+d[i]*v;jv[i]=c[i]+d[i]*u;}
}
static double diagonal(const double *b,const double *d){
 double dd=dot(d,d),t=dd?fmin(1.,fmax(0.,-dot(b,d)/dd)):0.,z[3];
 for(int i=0;i<3;i++)z[i]=b[i]+d[i]*t;
 return dot(z,z);
}
/* A strictly-convex, well-separated candidate fast path. Uncertain KKT,
   near-seam, folded or nonconverged queries return 0 to the Python oracle. */
int fast_convex_q1(const double *vertices,const double *point,double *out){
 double scale2=0.;
 for(int i=0;i<3;i++){double lo=vertices[i],hi=lo;
  for(int j=1;j<4;j++){lo=fmin(lo,vertices[3*j+i]);hi=fmax(hi,vertices[3*j+i]);}
  scale2+=(hi-lo)*(hi-lo);}
 double scale=sqrt(scale2);if(!(scale>0.)||!isfinite(scale))return 0;
 double x[12],a[3],b[3],c[3],d[3];
 for(int j=0;j<4;j++)for(int i=0;i<3;i++)x[3*j+i]=(vertices[3*j+i]-vertices[i])/scale;
 for(int i=0;i<3;i++){a[i]=(vertices[i]-point[i])/scale;b[i]=x[3+i];c[i]=x[9+i];d[i]=x[6+i]-b[i]-c[i];}
 double off=0.;
 for(int u=0;u<2;u++)for(int v=0;v<2;v++)
  off=fmax(off,fabs(dot(b,c)+dot(a,d)+2*u*dot(b,d)+2*v*dot(c,d)+2*u*v*dot(d,d)));
 double curvature=fmin(diagonal(b,d),diagonal(c,d))-off;
 if(!(curvature>1e-8))return 0;
 double u=.5,v=.5,z[3],ju[3],jv[3];int converged=0;
 for(int iteration=0;iteration<20;iteration++){
  delta(a,b,c,d,u,v,z,ju,jv);double g0=dot(z,ju),g1=dot(z,jv);
  if(hypot(g0,g1)<16.*DBL_EPSILON*curvature){converged=1;break;}
  double h00=dot(ju,ju),h11=dot(jv,jv),h01=dot(ju,jv)+dot(z,d),det=h00*h11-h01*h01;
  if(!(det>1e-16))break;
  double du=(h11*g0-h01*g1)/det,dv=(h00*g1-h01*g0)/det;
  u-=du;v-=dv;
  if(u<-.1||u>1.1||v<-.1||v>1.1)break;
 }
 double best=INFINITY,bu=0.,bv=0.;
 double interior_margin=1e-8;
#ifdef Q1_ADAPTIVE_MARGIN
 double coordinate=0.;for(int j=0;j<12;j++)coordinate=fmax(coordinate,fabs(vertices[j]));
 for(int j=0;j<3;j++)coordinate=fmax(coordinate,fabs(point[j]));
 double rounding=64.*DBL_EPSILON*fmax(1.,coordinate/scale)*(1.+sqrt(dot(a,a)))
                 *(1.+sqrt(dot(b,b))+sqrt(dot(c,c))+sqrt(dot(d,d)));
 delta(a,b,c,d,u,v,z,ju,jv);
 /* Strong convexity bounds distance from this stationary approximation to
  * the constrained minimizer. Translation/scale-conditioned roundoff is
  * included instead of admitting all points within a fixed seam margin. */
 interior_margin=16.*(hypot(dot(z,ju),dot(z,jv))+rounding)/curvature;
#endif
 if(converged&&u>interior_margin&&u<1.-interior_margin&&v>interior_margin&&v<1.-interior_margin){
  delta(a,b,c,d,u,v,z,ju,jv);best=dot(z,z);bu=u;bv=v;
 }
 const double corners[8]={0.,0.,1.,0.,1.,1.,0.,1.};
 for(int e=0;e<4;e++){
  int next=(e+1)%4;double z0[3],z1[3],edge[3],dummy[3],dummy2[3];
  delta(a,b,c,d,corners[2*e],corners[2*e+1],z0,dummy,dummy2);
  delta(a,b,c,d,corners[2*next],corners[2*next+1],z1,dummy,dummy2);
  for(int i=0;i<3;i++)edge[i]=z1[i]-z0[i];
  double norm=dot(edge,edge);if(!(norm>0.))return 0;
  double r=fmin(1.,fmax(0.,-dot(z0,edge)/norm));
  u=corners[2*e]+r*(corners[2*next]-corners[2*e]);
  v=corners[2*e+1]+r*(corners[2*next+1]-corners[2*e+1]);
  delta(a,b,c,d,u,v,z,ju,jv);double g[2]={dot(z,ju),dot(z,jv)},uv[2]={u,v};int valid=1;
  for(int k=0;k<2;k++){
   if(uv[k]==0.)valid&=g[k]>1e-10;
   else if(uv[k]==1.)valid&=g[k]<-1e-10;
   else valid&=fabs(g[k])<2e-15&&uv[k]>1e-8&&uv[k]<1.-1e-8;
  }
  double value=dot(z,z);if(valid&&value<best){best=value;bu=u;bv=v;}
 }
 if(!isfinite(best))return 0;
 delta(a,b,c,d,bu,bv,z,ju,jv);
 out[0]=bu;out[1]=bv;
 out[2]=(1-bu)*(1-bv);out[3]=bu*(1-bv);out[4]=bu*bv;out[5]=(1-bu)*bv;
 for(int i=0;i<3;i++){double position=0.;for(int j=0;j<4;j++)position+=out[2+j]*x[3*j+i];out[6+i]=vertices[i]+scale*position;}
 double normal[3]={ju[1]*jv[2]-ju[2]*jv[1],ju[2]*jv[0]-ju[0]*jv[2],ju[0]*jv[1]-ju[1]*jv[0]};
 double length=sqrt(dot(normal,normal));if(!(length>0.))return 0;
 for(int i=0;i<3;i++)out[9+i]=normal[i]/length;
 double distance[3];for(int i=0;i<3;i++)distance[i]=point[i]-out[6+i];
 out[12]=sqrt(dot(distance,distance));return 1;
}
