/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Internal C ABI. The Python adapter validates arrays and batch extents.
 */
void hex_gradient(int n,const double*x,const double*u,const double*g,const double*radius,
 const int*edges,double*f,int*admitted) {
 for(int k=0;k<n;k++) {
  const double*uk=u+24*k,*gk=g+24*k;double*fk=f+9*k;
  for(int i=0;i<3;i++)for(int j=0;j<3;j++){
   double value=0;for(int a=0;a<8;a++)value+=uk[3*a+i]*gk[3*a+j];
   fk[3*i+j]=value+(i==j);
  }
  double upper=0,scale_x=0,scale_u=0;
  for(int d=0;d<3;d++){double maximum=0;
   for(int a=0;a<4;a++){double square=0;
    int i0=edges[d*8+a*2],i1=edges[d*8+a*2+1];
    for(int j=0;j<3;j++){double v=uk[3*i1+j]-uk[3*i0+j];square+=v*v;}
    maximum=fmax(maximum,square);
   }upper+=maximum;
  }
  for(int j=0;j<24;j++){scale_x=fmax(scale_x,fabs(x[24*k+j]));scale_u=fmax(scale_u,fabs(uk[j]));}
  double bound=sqrt(upper)+256*DBL_EPSILON*(scale_x+scale_u);
  admitted[k]=isfinite(bound)&&bound<radius[k];
 }
}

void hex_bulk_response(int n,const double*f,const double*g,const double*v,const double*m,
 const double*volume,const double*rho,const double*speed,const double*length,
 const double*linear,const double*quadratic,double*force,double*power,double*rate,
 double*pressure,double*bound,int*status) {
 for(int k=0;k<n;k++) {
  const double*fk=f+9*k;double c[9],grad[24],scale=0;
  body_cofactor(fk,c);double det=(fk[0]*c[0]+fk[3]*c[3])+fk[6]*c[6];
  for(int j=0;j<9;j++)scale=fmax(scale,fabs(fk[j]));
  status[k]=0;
  if(!(det>1e-9*scale*scale*scale)){status[k]=1;continue;}
  rate[k]=0;bound[k]=0;
  for(int a=0;a<8;a++){double square=0;
   for(int i=0;i<3;i++){double value=0;
    for(int j=0;j<3;j++)value+=c[3*i+j]*g[24*k+3*a+j];
    grad[3*a+i]=value/det;rate[k]+=grad[3*a+i]*v[24*k+3*a+i];
    square+=grad[3*a+i]*grad[3*a+i];
   }bound[k]+=square/m[8*k+a];
  }
  double compression=fmin(rate[k],0),ql=quadratic[k]*length[k];
  pressure[k]=rho[k]*linear[k]*speed[k]*length[k]*rate[k]-rho[k]*(ql*compression)*(ql*compression);
  double eta=rho[k]*linear[k]*speed[k]*length[k]-2*rho[k]*ql*ql*compression;
  double current=volume[k]*det;
  for(int j=0;j<24;j++)force[24*k+j]=current*pressure[k]*grad[j];
  power[k]=current*pressure[k]*rate[k];bound[k]*=current*eta;
 }
}
