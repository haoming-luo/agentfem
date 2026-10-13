/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Generic Q1 arithmetic; uncertain predicates retain the reference fallback.
 */
/* GMP exact-convexity recovery; no change to contact tolerance.
 * Binary64 geometry is interpreted exactly. Newton iterates are witnesses;
 * the exact normal-cone residual / Hessian bound certifies their error. */
#include <gmp.h>
#include <math.h>
#ifndef Q1_EXACT_ITERATIONS
#define Q1_EXACT_ITERATIONS 12
#endif

static void dot(mpq_t out,mpq_t *a,mpq_t *b,mpq_t tmp){
 mpq_set_ui(out,0,1);
 for(int j=0;j<3;j++){mpq_mul(tmp,a[j],b[j]);mpq_add(out,out,tmp);}
}
static void clamp(mpq_t x){if(mpq_sgn(x)<0)mpq_set_ui(x,0,1);else if(mpq_cmp_ui(x,1,1)>0)mpq_set_ui(x,1,1);}

#ifdef Q1_BERNSTEIN_BOUND
/* Exact degree-(2,2) determinant enclosure. No sampling or tolerance. */
static void bernstein_modulus(mpq_t out,mpq_t*a,mpq_t*b,mpq_t*c,mpq_t*d){
 mpq_t aa[3],zz[3],off[2][2],power[3][3],tmp,prod,beta,lower,trace,high;
 for(int i=0;i<3;i++){mpq_init(aa[i]);mpq_init(zz[i]);for(int j=0;j<3;j++)mpq_init(power[i][j]);}
 for(int i=0;i<2;i++)for(int j=0;j<2;j++)mpq_init(off[i][j]);
 mpq_inits(tmp,prod,beta,lower,trace,high,NULL);mpq_set_ui(out,0,1);
 dot(aa[0],b,b,prod);dot(aa[1],b,d,prod);mpq_mul_2exp(aa[1],aa[1],1);dot(aa[2],d,d,prod);
 dot(zz[0],c,c,prod);dot(zz[1],c,d,prod);mpq_mul_2exp(zz[1],zz[1],1);mpq_set(zz[2],aa[2]);
 dot(off[0][0],b,c,prod);dot(tmp,a,d,prod);mpq_add(off[0][0],off[0][0],tmp);
 mpq_set(off[1][0],aa[1]);mpq_set(off[0][1],zz[1]);mpq_mul_2exp(off[1][1],aa[2],1);
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)mpq_mul(power[i][j],zz[i],aa[j]);
 for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++)for(int l=0;l<2;l++){
  mpq_mul(tmp,off[i][j],off[k][l]);mpq_sub(power[i+k][j+l],power[i+k][j+l],tmp);
 }
 mpq_set(lower,power[0][0]);
 for(int i=0;i<3;i++)for(int j=0;j<3;j++){
  mpq_set_ui(beta,0,1);
  for(int k=0;k<=i;k++)for(int l=0;l<=j;l++){
   mpq_set(tmp,power[k][l]);
   if(i==1&&k==1)mpq_div_2exp(tmp,tmp,1);
   if(j==1&&l==1)mpq_div_2exp(tmp,tmp,1);
   mpq_add(beta,beta,tmp);
  }
  if(mpq_sgn(beta)<=0)goto clear;
  if(mpq_cmp(beta,lower)<0)mpq_set(lower,beta);
 }
 mpq_add(high,aa[0],aa[1]);mpq_add(high,high,aa[2]);
 mpq_set(trace,mpq_cmp(high,aa[0])>0?high:aa[0]);
 mpq_add(high,zz[0],zz[1]);mpq_add(high,high,zz[2]);
 mpq_add(trace,trace,mpq_cmp(high,zz[0])>0?high:zz[0]);
 if(mpq_sgn(trace)>0)mpq_div(out,lower,trace);
 clear:
 for(int i=0;i<3;i++){mpq_clear(aa[i]);mpq_clear(zz[i]);for(int j=0;j<3;j++)mpq_clear(power[i][j]);}
 for(int i=0;i<2;i++)for(int j=0;j<2;j++)mpq_clear(off[i][j]);
 mpq_clears(tmp,prod,beta,lower,trace,high,NULL);
}
#endif

static int exact_convex_attempt(const double*x,const double*p,double*out){
 mpq_t z[64];for(int i=0;i<64;i++)mpq_init(z[i]);
 mpq_t *a=z,*b=z+3,*c=z+6,*d=z+9,*delta=z+12,*du=z+15,*dv=z+18;
 /* Scratch/result slots are deliberately disjoint from dot's product. */
 #define DD z[21]
 #define BC z[22]
 #define AD z[23]
 #define BD z[24]
 #define CD z[25]
 #define OFF z[26]
 #define MOD z[27]
 #define R z[28]
 #define U z[29]
 #define V z[30]
 #define GU z[31]
 #define GV z[32]
 #define HUU z[33]
 #define HVV z[34]
 #define HUV z[35]
 #define DET z[36]
 #define T z[37]
 #define S z[38]
 #define PROD z[39]
 #define RU z[40]
 #define RV z[41]
 #define LIMIT z[42]
 #define UN z[43]
 #define VN z[44]
 #define MINB z[45]
 #define MINC z[46]
 int ok=-2;
 for(int j=0;j<3;j++){
  mpq_set_d(T,x[j]);mpq_set_d(S,p[j]);mpq_sub(a[j],T,S);
  mpq_set_d(b[j],x[3+j]);mpq_sub(b[j],b[j],T);
  mpq_set_d(c[j],x[9+j]);mpq_sub(c[j],c[j],T);
  mpq_set_d(d[j],x[6+j]);mpq_sub(d[j],d[j],T);mpq_sub(d[j],d[j],b[j]);mpq_sub(d[j],d[j],c[j]);
 }
 dot(DD,d,d,PROD);dot(BC,b,c,PROD);dot(AD,a,d,PROD);dot(BD,b,d,PROD);dot(CD,c,d,PROD);
 for(int k=0;k<2;k++){
  if(mpq_sgn(DD)){mpq_neg(R,k?CD:BD);mpq_div(R,R,DD);clamp(R);}else mpq_set_ui(R,0,1);
  for(int j=0;j<3;j++){mpq_mul(delta[j],R,d[j]);mpq_add(delta[j],delta[j],k?c[j]:b[j]);}
  dot(k?MINC:MINB,delta,delta,PROD);
 }
 mpq_set_ui(OFF,0,1);
 for(int i=0;i<2;i++)for(int j=0;j<2;j++){
  mpq_add(T,BC,AD);
  if(i){mpq_mul_2exp(S,BD,1);mpq_add(T,T,S);}
  if(j){mpq_mul_2exp(S,CD,1);mpq_add(T,T,S);}
  if(i&&j){mpq_mul_2exp(S,DD,1);mpq_add(T,T,S);}
  mpq_abs(T,T);if(mpq_cmp(T,OFF)>0)mpq_set(OFF,T);
 }
 mpq_sub(MOD,mpq_cmp(MINB,MINC)<0?MINB:MINC,OFF);
 mpq_mul(T,MINB,MINC);mpq_mul(S,OFF,OFF);mpq_sub(T,T,S);
 if(mpq_sgn(MINB)>0&&mpq_sgn(MINC)>0&&mpq_sgn(T)>0){
  mpq_add(S,MINB,MINC);mpq_div(T,T,S);
  if(mpq_cmp(T,MOD)>0)mpq_set(MOD,T);
 }
#ifdef Q1_BERNSTEIN_BOUND
 if(mpq_sgn(MOD)<=0)bernstein_modulus(MOD,a,b,c,d);
#endif
 if(mpq_sgn(MOD)<=0){ok=-1;goto done;}
 mpq_div_2exp(LIMIT,MOD,48);mpq_mul(LIMIT,LIMIT,LIMIT);
 mpq_set_ui(U,1,2);mpq_set_ui(V,1,2);
 for(int iteration=0;iteration<Q1_EXACT_ITERATIONS;iteration++){
  mpq_mul(R,U,V);
  for(int j=0;j<3;j++){
   mpq_mul(delta[j],b[j],U);mpq_add(delta[j],delta[j],a[j]);
   mpq_mul(T,c[j],V);mpq_add(delta[j],delta[j],T);
   mpq_mul(T,d[j],R);mpq_add(delta[j],delta[j],T);
   mpq_mul(du[j],d[j],V);mpq_add(du[j],du[j],b[j]);
   mpq_mul(dv[j],d[j],U);mpq_add(dv[j],dv[j],c[j]);
  }
  dot(GU,delta,du,PROD);dot(GV,delta,dv,PROD);
  mpq_set(RU,GU);mpq_set(RV,GV);
  if((mpq_sgn(U)==0&&mpq_sgn(GU)>=0)||(mpq_cmp_ui(U,1,1)==0&&mpq_sgn(GU)<=0))mpq_set_ui(RU,0,1);
  if((mpq_sgn(V)==0&&mpq_sgn(GV)>=0)||(mpq_cmp_ui(V,1,1)==0&&mpq_sgn(GV)<=0))mpq_set_ui(RV,0,1);
  mpq_mul(T,RU,RU);mpq_mul(S,RV,RV);mpq_add(T,T,S);
  if(mpq_cmp(T,LIMIT)<=0){out[0]=mpq_get_d(U);out[1]=mpq_get_d(V);ok=1;goto done;}
  dot(HUU,du,du,PROD);dot(HVV,dv,dv,PROD);dot(HUV,du,dv,PROD);dot(T,delta,d,PROD);mpq_add(HUV,HUV,T);
  mpq_mul(DET,HUU,HVV);mpq_mul(T,HUV,HUV);mpq_sub(DET,DET,T);
  if(mpq_sgn(DET)<=0)goto done;
  if((mpq_sgn(U)==0||mpq_cmp_ui(U,1,1)==0)&&mpq_sgn(RU)==0){
   mpq_set(UN,U);mpq_div(T,GV,HVV);mpq_sub(VN,V,T);
  }else if((mpq_sgn(V)==0||mpq_cmp_ui(V,1,1)==0)&&mpq_sgn(RV)==0){
   mpq_set(VN,V);mpq_div(T,GU,HUU);mpq_sub(UN,U,T);
  }else{
   mpq_mul(T,HVV,GU);mpq_mul(S,HUV,GV);mpq_sub(T,T,S);mpq_div(T,T,DET);mpq_sub(UN,U,T);
   mpq_mul(T,HUU,GV);mpq_mul(S,HUV,GU);mpq_sub(T,T,S);mpq_div(T,T,DET);mpq_sub(VN,V,T);
  }
  clamp(UN);clamp(VN);
  /* Rounding back bounds rational growth. Truncation direction does not
   * weaken the next exact residual test. */
  mpq_set_d(U,mpq_get_d(UN));mpq_set_d(V,mpq_get_d(VN));
 }
 done:
 for(int i=0;i<64;i++)mpq_clear(z[i]);return ok;
}

int exact_convex_parameters(const double*x,const double*p,double*out){
 return exact_convex_attempt(x,p,out)>0;
}
/* Diagnostic only: -1 unavailable convex bound, -2 unresolved witness. */
int exact_convex_failure_reason(const double*x,const double*p,double*out){
 return exact_convex_attempt(x,p,out);
}

extern int fast_convex_q1(const double*,const double*,double*);
int convex_q1(const double*x,const double*p,double*out){
 if(fast_convex_q1(x,p,out))return 1;
 if(!exact_convex_parameters(x,p,out))return 0;
 double u=out[0],v=out[1],a[3],b[3],d[3],du[3],dv[3];
 out[2]=(1-u)*(1-v);out[3]=u*(1-v);out[4]=u*v;out[5]=(1-u)*v;
 for(int j=0;j<3;j++){
  a[j]=x[3+j]-x[j];b[j]=x[9+j]-x[j];d[j]=(x[6+j]-x[9+j])-a[j];
  du[j]=a[j]+d[j]*v;dv[j]=b[j]+d[j]*u;
  out[6+j]=x[j]+u*a[j]+v*b[j]+u*v*d[j];
 }
 double normal[3]={du[1]*dv[2]-du[2]*dv[1],du[2]*dv[0]-du[0]*dv[2],du[0]*dv[1]-du[1]*dv[0]};
 double norm=hypot(hypot(normal[0],normal[1]),normal[2]);
 if(!isfinite(norm)||norm==0)return 0;
 for(int j=0;j<3;j++)out[9+j]=normal[j]/norm;
 out[12]=hypot(hypot(p[0]-out[6],p[1]-out[7]),p[2]-out[8]);
 return isfinite(out[12]);
}
