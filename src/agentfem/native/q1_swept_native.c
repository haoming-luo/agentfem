/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Generic Q1 arithmetic; uncertain predicates retain the reference fallback.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Sufficient separation proof only. Every dot-product operation is enclosed
 * with outward rounding; unresolved control hulls go to rational subdivision. */
static int filtered(const double *a,const double*b,const double*p,const double*q){
 double d[8][3],lo[8][3],hi[8][3];const int map[4]={0,3,1,2};
 for(int t=0;t<2;t++)for(int k=0;k<4;k++)for(int j=0;j<3;j++){
  int i=4*t+k;d[i][j]=(t?b:a)[3*map[k]+j]-(t?q:p)[j];
  lo[i][j]=nextafter(d[i][j],-INFINITY);hi[i][j]=nextafter(d[i][j],INFINITY);
  if(!isfinite(lo[i][j])||!isfinite(hi[i][j]))return 0;
 }
 for(int j=0;j<3;j++){
  int positive=1,negative=1;
  for(int i=0;i<8;i++){positive=positive&&(lo[i][j]>0);negative=negative&&(hi[i][j]<0);}
  if(positive||negative)return 1;
 }
 for(int dim=0;dim<3;dim++)for(int side=0;side<2;side++){
  int face[4],h=0;
  for(int i=0;i<8;i++)if(((i>>(2-dim))&1)==side)face[h++]=i;
  double v[3],w[3],axis[3],scale=0;
  for(int j=0;j<3;j++){v[j]=d[face[3]][j]-d[face[0]][j];w[j]=d[face[1]][j]-d[face[2]][j];}
  for(int j=0;j<3;j++){
   axis[j]=v[(j+1)%3]*w[(j+2)%3]-v[(j+2)%3]*w[(j+1)%3];
   scale=fmax(scale,fabs(axis[j]));
  }
  if(!isfinite(scale)||scale==0)continue;
  for(int j=0;j<3;j++)axis[j]/=scale;
  int positive=1,negative=1;
  for(int i=0;i<8;i++){
   double lower=0,upper=0;
   for(int j=0;j<3;j++){
    double x=lo[i][j]*axis[j],y=hi[i][j]*axis[j];
    if(!isfinite(x)||!isfinite(y)){positive=negative=0;break;}
    double l=nextafter(fmin(x,y),-INFINITY),u=nextafter(fmax(x,y),INFINITY);
    if(j==0){lower=l;upper=u;}
    else {lower=nextafter(lower+l,-INFINITY);upper=nextafter(upper+u,INFINITY);}
   }
   positive=positive&&(lower>0);negative=negative&&(upper<0);
   if(!positive&&!negative)break;
  }
  if(positive||negative)return 1;
 }
 return 0;
}

#ifdef Q1_INTERVAL_SUBDIVISION
#ifndef Q1_REFINE_DEPTH
#define Q1_REFINE_DEPTH 6
#endif
#ifndef Q1_REFINE_BOXES
#define Q1_REFINE_BOXES 127
#endif
static int control_separated(double lo[8][3],double hi[8][3]){
 for(int j=0;j<3;j++){int pos=1,neg=1;
  for(int i=0;i<8;i++){pos=pos&&(lo[i][j]>0);neg=neg&&(hi[i][j]<0);}
  if(pos||neg)return 1;
 }
 double center[8][3];
 for(int i=0;i<8;i++)for(int j=0;j<3;j++)center[i][j]=lo[i][j]+.5*(hi[i][j]-lo[i][j]);
 for(int dim=0;dim<3;dim++)for(int side=0;side<2;side++){
  int face[4],h=0;for(int i=0;i<8;i++)if(((i>>(2-dim))&1)==side)face[h++]=i;
  double v[3],w[3],axis[3],scale=0;
  for(int j=0;j<3;j++){v[j]=center[face[3]][j]-center[face[0]][j];w[j]=center[face[1]][j]-center[face[2]][j];}
  for(int j=0;j<3;j++){axis[j]=v[(j+1)%3]*w[(j+2)%3]-v[(j+2)%3]*w[(j+1)%3];scale=fmax(scale,fabs(axis[j]));}
  if(!isfinite(scale)||scale==0)continue;
  for(int j=0;j<3;j++)axis[j]/=scale;
  int pos=1,neg=1;
  for(int i=0;i<8;i++){
   double l=0,u=0;
   for(int j=0;j<3;j++){
    double x=lo[i][j]*axis[j],y=hi[i][j]*axis[j];
    if(!isfinite(x)||!isfinite(y)){pos=neg=0;break;}
    double a=nextafter(fmin(x,y),-INFINITY),b=nextafter(fmax(x,y),INFINITY);
    if(j==0){l=a;u=b;}else{l=nextafter(l+a,-INFINITY);u=nextafter(u+b,INFINITY);}
   }
   pos=pos&&(l>0);neg=neg&&(u<0);if(!pos&&!neg)break;
  }
  if(pos||neg)return 1;
 }
 return 0;
}
static int refine(double lo[8][3],double hi[8][3],int depth,int *visited){
 if(++*visited>Q1_REFINE_BOXES)return 0;
 if(control_separated(lo,hi))return 1;
 if(depth==Q1_REFINE_DEPTH)return 0;
 int axis=0;double largest=-1;
 for(int d=0;d<3;d++){int bit=1<<(2-d);double span=0;
  for(int i=0;i<8;i++)if(!(i&bit))for(int j=0;j<3;j++)
   span=fmax(span,fabs((lo[i|bit][j]-lo[i][j])+(hi[i|bit][j]-hi[i][j])));
  if(span>largest){largest=span;axis=d;}
 }
 if(!isfinite(largest)||largest<=0)return 0;
 int bit=1<<(2-axis);double ll[8][3],lh[8][3],rl[8][3],rh[8][3];
 for(int i=0;i<8;i++)if(!(i&bit))for(int j=0;j<3;j++){
  int k=i|bit;
  double ml=nextafter(nextafter(lo[i][j]+lo[k][j],-INFINITY)*.5,-INFINITY);
  double mh=nextafter(nextafter(hi[i][j]+hi[k][j],INFINITY)*.5,INFINITY);
  if(!isfinite(ml)||!isfinite(mh))return 0;
  ll[i][j]=lo[i][j];lh[i][j]=hi[i][j];ll[k][j]=ml;lh[k][j]=mh;
  rl[i][j]=ml;rh[i][j]=mh;rl[k][j]=lo[k][j];rh[k][j]=hi[k][j];
 }
 return refine(ll,lh,depth+1,visited)&&refine(rl,rh,depth+1,visited);
}
static int refined_filter(const double*a,const double*b,const double*p,const double*q){
 double lo[8][3],hi[8][3];const int map[4]={0,3,1,2};int visited=0;
 for(int t=0;t<2;t++)for(int k=0;k<4;k++)for(int j=0;j<3;j++){
  double v=(t?b:a)[3*map[k]+j]-(t?q:p)[j];
  lo[4*t+k][j]=nextafter(v,-INFINITY);hi[4*t+k][j]=nextafter(v,INFINITY);
  if(!isfinite(lo[4*t+k][j])||!isfinite(hi[4*t+k][j]))return 0;
 }
 return refine(lo,hi,0,&visited);
}
#endif

int64_t swept_candidates(int64_t nq,int64_t nn,int64_t capacity,
 const int64_t*left,const int64_t*right,const int64_t*facet,
 const double*x0,const double*x1,const double*p0,const double*p1,
 double*lo,double*hi,int64_t*stack,int64_t*out){
 for(int64_t i=nn-1;i>=0;i--){
  int64_t f=facet[i];
  for(int j=0;j<3;j++){
   if(f>=0){double l=INFINITY,u=-INFINITY;
    for(int k=0;k<4;k++){l=fmin(l,fmin(x0[12*f+3*k+j],x1[12*f+3*k+j]));u=fmax(u,fmax(x0[12*f+3*k+j],x1[12*f+3*k+j]));}
    lo[3*i+j]=l;hi[3*i+j]=u;
   }else{lo[3*i+j]=fmin(lo[3*left[i]+j],lo[3*right[i]+j]);hi[3*i+j]=fmax(hi[3*left[i]+j],hi[3*right[i]+j]);}
  }
 }
 int64_t count=0;
 for(int64_t i=0;i<nq;i++){
  const double*a=p0+3*i,*b=p1+3*i;double l[3],u[3];
  for(int j=0;j<3;j++){l[j]=fmin(a[j],b[j]);u[j]=fmax(a[j],b[j]);}
  int64_t top=0;stack[top++]=0;
  while(top){int64_t node=stack[--top];int overlap=1;
   for(int j=0;j<3;j++)if(lo[3*node+j]>u[j]||hi[3*node+j]<l[j]){overlap=0;break;}
   if(!overlap)continue;
   int64_t f=facet[node];
   if(f<0){stack[top++]=left[node];stack[top++]=right[node];}
   else if(!filtered(x0+12*f,x1+12*f,a,b)){
#ifdef Q1_INTERVAL_SUBDIVISION
    if(refined_filter(x0+12*f,x1+12*f,a,b))continue;
#endif
    if(count==capacity)return -1;
    out[2*count]=i;out[2*count+1]=f;count++;
   }
  }
 }
 return count;
}
