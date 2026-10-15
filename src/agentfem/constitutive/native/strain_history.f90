! SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
! SPDX-License-Identifier: Apache-2.0
! Generated dimensions/policy; no constitutive equation lives in this adapter.
module strain_history_policy
use iso_c_binding
implicit none
integer,parameter :: ns=@NS@, np=@NP@
real(c_double),parameter :: stretch_limit=@LIMIT@
end module

subroutine history_update(n,props,rho,dt,de,so,qo,eo,io,sn,qn,en,inn) bind(C)
 use strain_history_policy
 implicit none
 integer(c_int),value :: n
 real(c_double),intent(in) :: props(np),rho(n),dt,de(n,6),so(n,6),qo(n,ns),eo(n),io(n)
 real(c_double),intent(out) :: sn(n,6),qn(n,ns),en(n),inn(n)
 real(c_double) :: xyz(n,3),length(n),spin(n,3),temp(n),stretch(n,6),grad(n,9),field(n,1)
 character(kind=c_char) :: name(80)
 interface
 subroutine agentfem_vumat_batch(n,ns,np,nf,st,tt,dt,props,rho,xyz,length,de,spin, &
 to,uo,fo,fvo,so,qo,eo,io,tn,un,fn,fvn,sn,qn,en,inn,cname) bind(C)
 use iso_c_binding
 integer(c_int),value :: n,ns,np,nf
 real(c_double),value :: st,tt,dt
 real(c_double) :: props(np),rho(n),xyz(n,3),length(n),de(n,6),spin(n,3)
 real(c_double) :: to(n),uo(n,6),fo(n,9),fvo(n,nf),so(n,6),qo(n,ns),eo(n),io(n)
 real(c_double) :: tn(n),un(n,6),fn(n,9),fvn(n,nf),sn(n,6),qn(n,ns),en(n),inn(n)
 character(kind=c_char) :: cname(80)
 end subroutine
 end interface
 ! These channels are explicitly declared unused by the reviewed source.
 xyz=0; length=1; spin=0; temp=0; stretch=0; grad=0; field=0
 stretch(:,1:3)=1; grad(:,1:3)=1; name=' '
 call agentfem_vumat_batch(n,ns,np,1,dt,dt,dt,props,rho,xyz,length,de,spin, &
 temp,stretch,grad,field,so,qo,eo,io,temp,stretch,grad,field,sn,qn,en,inn,name)
end subroutine

subroutine fused_material(n,old,new,state,props,dt,rho0,stress,nextstate,work,status) bind(C)
 use native_polar_tools
 use strain_history_policy
 use, intrinsic :: ieee_arithmetic
 implicit none
 integer(c_int),value :: n
 real(c_double),intent(in) :: old(3,3,n),new(3,3,n),state(ns+6,n),props(np)
 real(c_double),value :: dt,rho0
 real(c_double),intent(out) :: stress(3,3,n),nextstate(ns+6,n),work(n)
 integer(c_int),intent(out) :: status(n)
 real(c_double) :: strain(3,3,n),r0(3,3,n),r1(3,3,n)
 real(c_double) :: de(n,6),so(n,6),qo(n,ns),sn(n,6),qn(n,ns),en(n),inn(n),rho(n),zero(n)
 real(c_double) :: s(3,3),rotated(3,3),jmid
 integer :: k
 interface
 subroutine polar_increment(n,old,new,strain,r0,r1,status) bind(C)
 use iso_c_binding
 integer(c_int),value::n
 real(c_double),intent(in)::old(3,3,n),new(3,3,n)
 real(c_double),intent(out)::strain(3,3,n),r0(3,3,n),r1(3,3,n)
 integer(c_int),intent(out)::status(n)
 end subroutine
 subroutine history_update(n,props,rho,dt,de,so,qo,eo,io,sn,qn,en,inn) bind(C)
 use strain_history_policy
 integer(c_int),value::n
 real(c_double),intent(in)::props(np),rho(n),dt,de(n,6),so(n,6),qo(n,ns),eo(n),io(n)
 real(c_double),intent(out)::sn(n,6),qn(n,ns),en(n),inn(n)
 end subroutine
 end interface
 stress=ieee_value(0d0,ieee_quiet_nan); nextstate=ieee_value(0d0,ieee_quiet_nan)
 work=ieee_value(0d0,ieee_quiet_nan)
 sn=ieee_value(0d0,ieee_quiet_nan); qn=ieee_value(0d0,ieee_quiet_nan)
 en=ieee_value(0d0,ieee_quiet_nan); inn=ieee_value(0d0,ieee_quiet_nan)
 call polar_increment(n,old,new,strain,r0,r1,status)
 do k=1,n
  if(status(k)/=0) then
   strain(:,:,k)=0d0; r1(:,:,k)=ident()
  endif
  de(k,:)=[strain(1,1,k),strain(2,2,k),strain(3,3,k),strain(1,2,k),strain(2,3,k),strain(1,3,k)]
  so(k,:)=state(ns+1:ns+6,k); qo(k,:)=state(1:ns,k)
  jmid=(det(old(:,:,k))+det(new(:,:,k)))/2
  if(jmid<=0.or..not.ieee_is_finite(jmid)) then
   status(k)=2; jmid=1d0
  endif
  rho(k)=rho0/jmid
 enddo
 zero=0d0
 call history_update(n,props,rho,dt,de,so,qo,zero,zero,sn,qn,en,inn)
 do k=1,n
  s(1,:)=[sn(k,1),sn(k,4),sn(k,6)]
  s(2,:)=[sn(k,4),sn(k,2),sn(k,5)]
  s(3,:)=[sn(k,6),sn(k,5),sn(k,3)]
  rotated=matmul(matmul(transpose(r1(:,:,k)),s),r1(:,:,k))
  stress(:,:,k)=transpose(rotated)
  nextstate(1:ns,k)=qn(k,:); nextstate(ns+1:ns+6,k)=sn(k,:)
  work(k)=rho0*en(k)
 enddo
end subroutine

subroutine history_state_screen(n,state,status) bind(C)
use strain_history_policy
use, intrinsic :: ieee_arithmetic
implicit none
integer(c_int),value :: n
real(c_double),intent(in) :: state(ns+6,n)
integer(c_int),intent(out) :: status(n)
integer :: p
do p=1,n
 status(p)=1
 @BOUNDS@
enddo
end subroutine

subroutine history_increment_screen(n,old,new,status) bind(C)
use strain_history_policy
use, intrinsic :: ieee_arithmetic
implicit none
integer(c_int),value :: n
real(c_double),intent(in) :: old(9,n),new(9,n)
integer(c_int),intent(out) :: status(n)
integer :: p,i,j,k
real(c_double) :: gram(3,3),lower,delta,scale,rowerr,bound,term
do p=1,n
 delta=0; scale=1
 do i=1,9
  delta=delta+(new(i,p)-old(i,p))**2
  scale=max(scale,abs(old(i,p)),abs(new(i,p)))
 enddo
 bound=0
 do i=1,3
  rowerr=0
  do j=1,3
   gram(i,j)=0
   do k=1,3
    term=old(3*(k-1)+i,p)*old(3*(k-1)+j,p)
    gram(i,j)=gram(i,j)+term; rowerr=rowerr+abs(term)
   enddo
  enddo
  bound=max(bound,rowerr)
 enddo
 lower=huge(lower)
 do i=1,3
  lower=min(lower,2*gram(i,i)-sum(abs(gram(i,:))))
 enddo
 lower=lower-256*epsilon(lower)*bound
 status(p)=0
 if(ieee_is_finite(lower).and.ieee_is_finite(delta).and.lower>0) then
  if(sqrt(delta)+128*epsilon(delta)*scale < stretch_limit*sqrt(lower)) status(p)=1
 endif
enddo
end subroutine
