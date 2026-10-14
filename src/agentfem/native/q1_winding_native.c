/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Generic Q1 arithmetic; uncertain predicates retain the reference fallback.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int64_t segment_search(int64_t,int64_t,const double*,const double*,const int64_t*,const int64_t*,const int64_t*,const double*,const double*,double,int64_t*,int64_t*);
int64_t ray_hits(int64_t,const double*,const double*,const double*,double*);
int64_t interval_ray_hits(int64_t,const double*,const double*,const double*,double*);
static int order(const void *a,const void *b){int64_t x=*(const int64_t*)a,y=*(const int64_t*)b;return (x>y)-(x<y);}

/* Serial workspace reuse, same ordered rays and candidate equations. Any
 * unresolved ray sends the whole query back to the exact Python oracle. */
int winding_batch_n(int64_t nq,int64_t nn,int64_t root,int64_t nf,int64_t nd,
 const double *lo,const double *hi,const int64_t *left,const int64_t *right,
 const int64_t *facet,const double *vertices,const double *points,
 const double *directions,const double *lengths,double pad,int64_t *answers){
 const int64_t capacity=4096;
 int64_t *stack=malloc(nn*sizeof(int64_t)),*found=malloc(nn*sizeof(int64_t));
 double *patches=malloc(capacity*12*sizeof(double)),*hits=malloc(capacity*10*sizeof(double));
 if(!stack||!found||!patches||!hits){free(stack);free(found);free(patches);free(hits);return -1;}
 for(int64_t i=0;i<nq;i++){
  answers[i]=-1;int previous=-1;
  for(int k=0;k<nd;k++){
   const double *p=points+3*i,*r=directions+3*k;double delta[3];
   for(int j=0;j<3;j++){double end=p[j]+lengths[i]*r[j];delta[j]=end-p[j];}
   int64_t n=segment_search(nn,root,lo,hi,left,right,facet,p,delta,pad,stack,found);
   if(n<0)break;
   for(int64_t j=0;j<n;j++)found[j]/=2;
   qsort(found,n,sizeof(int64_t),order);
   int64_t unique=0;int invalid=0;
   for(int64_t j=0;j<n;j++){
    if(j && found[j]==found[j-1])continue;
    if(found[j]<0||found[j]>=nf||unique==capacity){invalid=1;break;}
    memcpy(patches+12*unique++,vertices+12*found[j],12*sizeof(double));
   }
   if(invalid)break;
   int64_t count=ray_hits(unique,patches,p,r,hits);
   if(count<0)count=interval_ray_hits(unique,patches,p,r,hits);
   /* An unproved direction supplies no vote. Try another deterministic ray;
    * invalid winding or two conflicting proved votes still require oracle. */
   if(count<0)continue;
   int value=0;for(int64_t j=0;j<count;j++)value+=(int)hits[5*j+4];
   if(value!=0&&value!=1)break;
   if(previous<0)previous=value;
   else {if(value==previous)answers[i]=value;break;}
  }
 }
 free(stack);free(found);free(patches);free(hits);return 0;
}

int winding_batch(int64_t nq,int64_t nn,int64_t root,int64_t nf,
 const double *lo,const double *hi,const int64_t *left,const int64_t *right,
 const int64_t *facet,const double *vertices,const double *points,
 const double *directions,const double *lengths,double pad,int64_t *answers){
 return winding_batch_n(nq,nn,root,nf,2,lo,hi,left,right,facet,vertices,points,
                         directions,lengths,pad,answers);
}
