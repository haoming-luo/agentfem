/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * The existing conservative affine-remainder screen, shared by standalone
 * admission and the fused gradient path. False retains SVD/Bernstein checks.
 */
static int hex_affine_admitted(const double *x, const double *u, const double *f,
                               double radius, const int *edges) {
    double d[9],g[9],rem[24],orient=0.,rho=0.,gnorm=0.,xs=0.,us=0.;
    int valid=isfinite(radius) && radius>0.;
    for (int i=0;i<9;++i) {d[i]=f[i]-(i%4==0 ? 1.:0.);valid &= isfinite(f[i]);}
    for (int i=0;i<24;++i) {
        valid &= isfinite(x[i]) && isfinite(u[i]);
        xs=fmax(xs,fabs(x[i]));us=fmax(us,fabs(u[i]));
    }
    if (!valid) return 0;
    for(int i=0;i<3;++i) {
        double row=0.;for(int j=0;j<3;++j)row+=fabs(d[3*i+j]);orient=fmax(orient,row);
    }
    /* A strict near-identity homotopy proves orientation, not Gram alone. */
    if (!isfinite(orient) || orient>=.5) return 0;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
        double value=0.;for(int l=0;l<3;++l)value+=f[3*l+i]*f[3*l+j];g[3*i+j]=value;
    }
    for(int i=0;i<3;++i) {
        double row=0.,delta=0.;for(int j=0;j<3;++j) {
            row+=fabs(g[3*i+j]);delta+=fabs(g[3*i+j]-(i==j ? 1.:0.));
        }
        gnorm=fmax(gnorm,row);rho=fmax(rho,delta);
    }
    rho+=256.*DBL_EPSILON*(1.+gnorm);
    if (!isfinite(rho) || rho>=.5) return 0;
    for(int a=0;a<8;++a)for(int i=0;i<3;++i) {
        double value=0.;for(int j=0;j<3;++j)value+=d[3*i+j]*(x[3*a+j]-x[j]);
        rem[3*a+i]=(u[3*a+i]-u[i])-value;
    }
    double bound=0.;
    for(int axis=0;axis<3;++axis) {
        double maximum=0.;
        for(int edge=0;edge<4;++edge) {
            int a=edges[8*axis+2*edge],b=edges[8*axis+2*edge+1];double squared=0.;
            for(int i=0;i<3;++i) {double delta=rem[3*b+i]-rem[3*a+i];squared+=delta*delta;}
            if (!isfinite(squared)) valid=0;
            maximum=fmax(maximum,squared);
        }
        bound+=maximum;
    }
    double guard=512.*DBL_EPSILON*(xs*(1.+sqrt(1.+rho))+us);
    double total=sqrt(bound)+guard;
    return valid && isfinite(total) && total<radius*sqrt(1.-rho);
}
