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
    /* Gram alone cannot distinguish a rotation from a reflection. Retain the
     * cheap near-identity homotopy; outside that ball use a filtered determinant
     * sign, NOT an unguarded determinant or a mean-J cell-validity shortcut.
     * Normalize before products, leaving cancellation/subnormal cases to the
     * established LU/SVD/Bernstein fallback. The final affine-remainder bound
     * below still has to certify the entire cell. */
    if (!isfinite(orient)) return 0;
    if (orient>=.5) {
        double scale=0., a[9];
        for(int i=0;i<9;++i) scale=fmax(scale,fabs(f[i]));
        if (!(scale>0.) || !isfinite(scale)) return 0;
        for(int i=0;i<9;++i) a[i]=f[i]/scale;
        double terms[6]={a[0]*a[4]*a[8],a[1]*a[5]*a[6],a[2]*a[3]*a[7],
                         a[2]*a[4]*a[6],a[1]*a[3]*a[8],a[0]*a[5]*a[7]};
        double det=0.,permanent=0.;
        for(int i=0;i<6;++i) {
            det += i<3 ? terms[i] : -terms[i];
            permanent += fabs(terms[i]);
        }
        double error=64.*DBL_EPSILON*permanent+64.*DBL_MIN;
        if (!isfinite(det) || !(det>error)) return 0;
    }
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
        double value=0.;for(int l=0;l<3;++l)value+=f[3*l+i]*f[3*l+j];g[3*i+j]=value;
    }
    for(int i=0;i<3;++i) {
        double row=0.,delta=0.;for(int j=0;j<3;++j) {
            row+=fabs(g[3*i+j]);delta+=fabs(g[3*i+j]-(i==j ? 1.:0.));
        }
        gnorm=fmax(gnorm,row);rho=fmax(rho,delta);
    }
    double gram_error=256.*DBL_EPSILON*(1.+gnorm);
    rho+=gram_error;
    if (!isfinite(rho) || !isfinite(gnorm)) return 0;
    double lower=1.-rho,upper=1.+rho;
    if (rho>=.5) {
        /* Gershgorin bounds do not require stretches near one. They are
         * sufficient singular-value bounds, not estimates: when inconclusive,
         * retain SVD. Separate positive orientation was proved above. */
        lower=DBL_MAX;
        for(int i=0;i<3;++i) {
            double off=0.;
            for(int j=0;j<3;++j) if(i!=j) off+=fabs(g[3*i+j]);
            lower=fmin(lower,g[4*i]-off);
        }
        lower-=gram_error;upper=gnorm+gram_error;
        if (!(lower>0.) || !isfinite(upper)) return 0;
    }
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
    double guard=512.*DBL_EPSILON*(xs*(1.+sqrt(upper))+us);
    double total=sqrt(bound)+guard;
    return valid && isfinite(total) && total<radius*sqrt(lower);
}
