/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Generic Q1 arithmetic; uncertain predicates retain the reference fallback.
 */
#include <stdint.h>
#include <math.h>
#include <float.h>

/* Same padded closed-segment slab test and traversal as the Python oracle.
   No triangle intersection, geometry approximation or fast-math permitted. */
int64_t segment_search(int64_t n, int64_t root, const double *lo,
  const double *hi, const int64_t *left, const int64_t *right,
  const int64_t *facet, const double *start, const double *delta, double pad,
  int64_t *stack, int64_t *found) {
  int64_t top=0, count=0;
  if (root < 0 || root >= n) return -1;
  stack[top++]=root;
  while (top) {
    int64_t node=stack[--top];
    if (node < 0 || node >= n) return -1;
    double near=0., far=1.;
    for (int j=0;j<3;j++) {
      double lower=lo[3*node+j]-pad, upper=hi[3*node+j]+pad;
      double origin=start[j], direction=delta[j];
      if (direction==0.) {
        if (origin<lower || origin>upper) {near=1.;far=0.;break;}
        continue;
      }
      double a=(lower-origin)/direction, b=(upper-origin)/direction;
      if (a>b) {double swap=a;a=b;b=swap;}
      if (isfinite(a)) a-=16.*DBL_EPSILON*fmax(1.,fabs(a));
      if (isfinite(b)) b+=16.*DBL_EPSILON*fmax(1.,fabs(b));
      near=fmax(near,a);far=fmin(far,b);
      if (near>far) break;
    }
    if (near>far) continue;
    if (facet[node]>=0) {
      if(count>=n) return -1;
      found[count++]=facet[node];
    } else {
      if(top+2>n) return -1;
      stack[top++]=left[node];stack[top++]=right[node];
    }
  }
  return count;
}

int64_t box_search(int64_t n,int64_t root,const double *lo,const double *hi,
 const int64_t *left,const int64_t *right,const int64_t *facet,
 const double *lower,const double *upper,int64_t *stack,int64_t *found){
 int64_t top=0,count=0;if(root<0||root>=n)return -1;stack[top++]=root;
 while(top){int64_t node=stack[--top];if(node<0||node>=n)return -1;
  if(hi[3*node]<lower[0]||hi[3*node+1]<lower[1]||hi[3*node+2]<lower[2]||
     lo[3*node]>upper[0]||lo[3*node+1]>upper[1]||lo[3*node+2]>upper[2])continue;
  if(facet[node]>=0){if(count>=n)return -1;found[count++]=facet[node];}
  else{if(top+2>n)return -1;stack[top++]=left[node];stack[top++]=right[node];}
 }
 return count;
}
