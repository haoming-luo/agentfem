! SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
! SPDX-License-Identifier: Apache-2.0
! Guarded linear-right-stretch path integral; status != 0 requires reference repair.
module native_polar_tools
 use iso_c_binding
 implicit none
 contains
 function ident() result(a)
 real(c_double) :: a(3,3)
 integer :: j
 a=0
 do j=1,3
 a(j,j)=1
 enddo
 end function
 function norm(a) result(v)
 real(c_double),intent(in):: a(3,3)
 real(c_double)::v
 v=sqrt(sum(a*a))
 end function
 function det(a) result(v)
 real(c_double),intent(in)::a(3,3)
 real(c_double)::v
 v=a(1,1)*(a(2,2)*a(3,3)-a(2,3)*a(3,2)) &
  -a(1,2)*(a(2,1)*a(3,3)-a(2,3)*a(3,1)) &
  +a(1,3)*(a(2,1)*a(3,2)-a(2,2)*a(3,1))
 end function
 subroutine polar(f,r,u,ok)
 real(c_double),intent(in)::f(3,3)
 real(c_double),intent(out)::r(3,3),u(3,3)
 logical,intent(out)::ok
 real(c_double)::e(3,3),eye(3,3),tol
 integer::j
 eye=ident(); tol=16*epsilon(1.d0)
 ok=.false.
 if(norm(matmul(transpose(f),f)-eye)>=.75d0.or.det(f)<=0) return
 r=f
 do j=1,7
 e=matmul(transpose(r),r)-eye
 if(norm(e)<=tol) exit
 r=matmul(r,eye-.5d0*e)
 enddo
 u=matmul(transpose(r),f); u=.5d0*(u+transpose(u))
 ok=norm(matmul(transpose(r),r)-eye)<=tol.and.norm(matmul(r,u)-f)<=tol
 end subroutine
 subroutine roots(u,y,z,ok)
 real(c_double),intent(in)::u(3,3)
 real(c_double),intent(out)::y(3,3),z(3,3)
 logical,intent(out)::ok
 real(c_double)::e(3,3),eye(3,3),tol
 integer::j
 eye=ident(); tol=16*epsilon(1.d0);y=u;z=eye
 do j=1,7
 e=matmul(z,y)-eye
 if(norm(e)<=tol) exit
 e=eye-.5d0*e;y=matmul(y,e);z=matmul(e,z)
 enddo
 ok=norm(matmul(y,y)-u)<=tol.and.norm(matmul(z,y)-eye)<=tol
 end subroutine
 subroutine symlogdelta(a,l,ok)
 real(c_double),intent(in)::a(3,3)
 real(c_double),intent(out)::l(3,3)
 logical,intent(out)::ok
 real(c_double)::power(3,3),remainder,radius
 real(c_double)::b(3,3),v(3,3),theta,c,s,app,aqq,apq,t1,t2,x,y,term,tol
 integer::k,p,q,i,j
 b=.5d0*(a+transpose(a));v=ident();tol=8*epsilon(1.d0)*norm(b)
 radius=norm(b)
 if(radius<.05d0) then
 l=b;power=b
 do j=2,16
 power=matmul(power,b)
 if(mod(j,2)==0) then
 l=l-power/j
 else
 l=l+power/j
 endif
 remainder=radius**(j+1)/((j+1)*(1-radius))
 if(remainder<=epsilon(1.d0)*max(radius,tiny(1.d0))) then
 l=.5d0*(l+transpose(l));ok=.true.;return
 endif
 enddo
 endif
 do k=1,40
 p=1;q=2
 if(abs(b(1,3))>abs(b(p,q))) then
 p=1;q=3
 endif
 if(abs(b(2,3))>abs(b(p,q))) then
 p=2;q=3
 endif
 if(abs(b(p,q))<=tol) exit
 app=b(p,p);aqq=b(q,q);apq=b(p,q)
 theta=.5d0*atan2(2*apq,aqq-app);c=cos(theta);s=sin(theta)
 do i=1,3
 if(i/=p.and.i/=q) then
 t1=b(i,p);t2=b(i,q)
 b(i,p)=c*t1-s*t2;b(p,i)=b(i,p)
 b(i,q)=s*t1+c*t2;b(q,i)=b(i,q)
 endif
 t1=v(i,p);t2=v(i,q);v(i,p)=c*t1-s*t2;v(i,q)=s*t1+c*t2
 enddo
 b(p,p)=c*c*app-2*c*s*apq+s*s*aqq
 b(q,q)=s*s*app+2*c*s*apq+c*c*aqq;b(p,q)=0;b(q,p)=0
 enddo
 ok=k<=40
 l=0
 do i=1,3
 x=b(i,i)
 if(x<=-1) then
 ok=.false.;return
 endif
 if(abs(x)<.01d0) then
 y=x;term=x
 do j=2,12
 term=-term*x;y=y+term/j
 enddo
 else
 y=log(1+x)
 endif
 do j=1,3
 l(:,j)=l(:,j)+y*v(:,i)*v(j,i)
 enddo
 enddo
 end subroutine
end module

subroutine polar_increment(n,old,new,strain,frame0,frame1,status) bind(C)
 use native_polar_tools
 implicit none
 integer(c_int),value::n
 real(c_double),intent(in)::old(3,3,n),new(3,3,n)
 real(c_double),intent(out)::strain(3,3,n),frame0(3,3,n),frame1(3,3,n)
 integer(c_int),intent(out)::status(n)
 real(c_double)::f0(3,3),f1(3,3),r0(3,3),r1(3,3),u0(3,3),u1(3,3)
 real(c_double)::root(3,3),invroot(3,3),delta(3,3),logarithm(3,3),integral(3,3)
 logical::ok
 integer::i
 status=1;strain=0;frame0=0;frame1=0
 do i=1,n
 f0=transpose(old(:,:,i));f1=transpose(new(:,:,i))
 call polar(f0,r0,u0,ok)
 if(.not.ok) cycle
 call polar(f1,r1,u1,ok)
 if(.not.ok) cycle
 call roots(u0,root,invroot,ok)
 if(.not.ok) cycle
 delta=matmul(matmul(invroot,u1-u0),invroot)
 call symlogdelta(delta,logarithm,ok)
 if(.not.ok) cycle
 integral=matmul(matmul(root,logarithm),invroot)
 strain(:,:,i)=.5d0*(integral+transpose(integral))
 frame0(:,:,i)=transpose(r0);frame1(:,:,i)=transpose(r1);status(i)=0
 enddo
end subroutine
