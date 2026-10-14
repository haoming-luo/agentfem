/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Generic Q1 arithmetic; uncertain predicates retain the reference fallback.
 */
#include <stdint.h>
#include <stdlib.h>
extern int64_t box_search(int64_t,int64_t,const double*,const double*,const int64_t*,const int64_t*,const int64_t*,const double*,const double*,int64_t*,int64_t*);
extern int convex_q1(const double*,const double*,double*);
static int compare(const void*a,const void*b){int64_t x=*(const int64_t*)a,y=*(const int64_t*)b;return (x>y)-(x<y);}
int batch_boxes(int64_t nq,int64_t nn,int64_t root,int64_t capacity,
 const double*points,double radius,const double*lo,const double*hi,
 const int64_t*left,const int64_t*right,const int64_t*facets,
 int64_t*stack,int64_t*scratch,int64_t*counts,int64_t*out){
 for(int64_t i=0;i<nq;i++){
  double lower[3],upper[3];for(int j=0;j<3;j++){lower[j]=points[3*i+j]-radius;upper[j]=points[3*i+j]+radius;}
  int64_t n=box_search(nn,root,lo,hi,left,right,facets,lower,upper,stack,scratch);if(n<0)return -1;
  for(int64_t k=0;k<n;k++)scratch[k]/=2;qsort(scratch,n,sizeof(int64_t),compare);
  int64_t count=0;
  for(int64_t k=0;k<n;k++)if(k==0||scratch[k]!=scratch[k-1]){if(count>=capacity)return -2;out[i*capacity+count++]=scratch[k];}
  counts[i]=count;
 }
 return 0;
}
void batch_projection(int64_t n,const double*x,const double*points,int64_t*status,double*out){
 for(int64_t i=0;i<n;i++)status[i]=convex_q1(x+12*i,points+3*i,out+13*i);
}
