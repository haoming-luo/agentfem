/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Generic Q1 arithmetic; uncertain predicates retain the reference fallback.
 */
#include <math.h>
#include <stdint.h>
#include <float.h>
static double dot(const double *a,const double *b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static void cross(const double*a,const double*b,double*c){c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];}
static double box(const double*p,const double*lo,const double*hi){double s=0;for(int j=0;j<3;j++){double z=fmax(0,fmax(lo[j]-p[j],p[j]-hi[j]));s+=z*z;}return sqrt(s);}
static double patch(const double*x,const double*p,double pad,double target,int depth){
 double lo[3],hi[3],a[3],b[3],normal[3],axes[15],offset[3];
 for(int j=0;j<3;j++){lo[j]=hi[j]=x[j];for(int h=1;h<4;h++){lo[j]=fmin(lo[j],x[3*h+j]);hi[j]=fmax(hi[j],x[3*h+j]);}a[j]=x[6+j]-x[j];b[j]=x[9+j]-x[3+j];offset[j]=p[j]-x[j];}
 cross(a,b,normal);double norm=sqrt(dot(normal,normal));if(norm)for(int j=0;j<3;j++)normal[j]/=norm;
 for(int j=0;j<3;j++)axes[j]=normal[j];
 for(int h=0;h<4;h++){for(int j=0;j<3;j++)a[j]=x[3*((h+1)%4)+j]-x[3*h+j];cross(a,normal,axes+3*(h+1));}
 double slab=0;
 for(int h=0;h<5;h++){double *axis=axes+3*h;norm=sqrt(dot(axis,axis));if(norm)for(int j=0;j<3;j++)axis[j]/=norm;
  double low=0,high=0;for(int v=1;v<4;v++){for(int j=0;j<3;j++)a[j]=x[3*v+j]-x[j];double y=dot(a,axis);low=fmin(low,y);high=fmax(high,y);}double y=dot(offset,axis);slab=fmax(slab,fmax(low-y,y-high));}
 double lower=fmax(0,fmax(box(p,lo,hi),slab)-pad);if(lower>=target||depth>=6)return lower;
 double grid[9][3]; /* corners 0..3, midpoints 4..7, center 8 */
 for(int j=0;j<3;j++){grid[8][j]=(x[j]+x[3+j]+x[6+j]+x[9+j])/4.;for(int h=0;h<4;h++){grid[h][j]=x[3*h+j];grid[h+4][j]=(x[3*h+j]+x[3*((h+1)%4)+j])/2.;}}
 const int ids[4][4]={{0,4,8,7},{4,1,5,8},{8,5,2,6},{7,8,6,3}};double best=INFINITY;
 for(int child=0;child<4;child++){double v[12];for(int h=0;h<4;h++)for(int j=0;j<3;j++)v[3*h+j]=grid[ids[child][h]][j];best=fmin(best,patch(v,p,pad,target,depth+1));if(best==0)break;}return best;
}
int distance_bounds(int64_t nq,int64_t nn,int64_t nf,int64_t root,
 const double*points,const double*targets,const double*x,double padding,double scale,
 const double*lo,const double*hi,const int64_t*left,const int64_t*right,const int64_t*facets,
 const double*blo,const double*bhi,const double*axes,const double*slo,const double*shi,
 int64_t*stack,double*pending,int64_t*visited,double*out){
 if(root<0||root>=nn)return -1;for(int64_t f=0;f<nf;f++)visited[f]=-1;
 for(int64_t i=0;i<nq;i++){const double*p=points+3*i;double pad=256*DBL_EPSILON*fmax(scale,fmax(fabs(p[0]),fmax(fabs(p[1]),fabs(p[2]))));
  int64_t top=1;stack[0]=root;pending[0]=0;double best=INFINITY;
  while(top){top--;int64_t node=stack[top];double bound=pending[top];if(bound>=best)continue;if(node<0||node>=nn)return -1;
   if(facets[node]>=0){int64_t f=facets[node]/2;if(f<0||f>=nf)return -1;if(visited[f]==i)continue;visited[f]=i;
    double delta[3];for(int j=0;j<3;j++)delta[j]=p[j]-x[12*f+j];double slab=0;
    for(int h=0;h<5;h++){double y=dot(axes+15*f+3*h,delta);slab=fmax(slab,fmax(slo[5*f+h]-y,y-shi[5*f+h]));}
    double leaf=fmax(0,fmax(box(p,blo+3*f,bhi+3*f),slab)-pad);if(leaf<targets[i])leaf=patch(x+12*f,p,pad,targets[i],0);best=fmin(best,leaf);
   }else{
    int64_t children[2]={left[node],right[node]};double b[2];
    for(int j=0;j<2;j++){if(children[j]<0||children[j]>=nn)return -1;b[j]=fmax(0,box(p,lo+3*children[j],hi+3*children[j])-padding-pad);}
    if(b[0]<b[1]){double t=b[0];b[0]=b[1];b[1]=t;int64_t k=children[0];children[0]=children[1];children[1]=k;}
    for(int j=0;j<2;j++)if(b[j]<best){if(top>=nn)return -1;stack[top]=children[j];pending[top++]=b[j];}
   }
  }
  if(!isfinite(best))return -1;out[i]=best;
 }
 return 0;
}
