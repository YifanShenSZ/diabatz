#include "mkl_rci.f90"

subroutine trust_region(fd, Jacobian, x, M, N, &
Warning, MaxIteration, MaxStepIteration, Precision, MinStepLength)
    use mkl_rci_type
    use mkl_rci
    implicit none

    external::fd, Jacobian
    integer,intent(in)::M,N
    real*8,dimension(N),intent(inout)::x
    logical,intent(in)::Warning
    integer,intent(in)::MaxIteration,MaxStepIteration
    real*8 ,intent(in)::Precision,MinStepLength
    !Reverse communication interface (RCI)
    integer::RCI_request!Recieve job request
    integer,dimension(6)::info!Results of input parameter checking
    type(handle_tr)::handle!Trust-region solver handle
    !Job control
    !tol(1:5) contains the stopping criteria for solving f'(x) = 0:
    !    1, trust region radius < tol(1)
    !    2, || f'(x) ||_2 < tol(2)
    !    3, || Jacobian ||_1 < tol(3) 
    !    4, || s ||_2 < tol(4), where s is the trial step
    !    5, || f'(x) ||_2 - || f'(x) - Jacobian . s ||_2 < tol(5)
    !tol(6) is the precision of s calculation
    real*8,dimension(6)::tol
    !TotalIteration harvests the solver stops after how many iterations
    !StopReason harvests why the solver has stopped:
    !    1,   max iteration exceeded
    !    2-6, tol(StopReason-1) is met
    integer::TotalIteration,StopReason
    real*8::InitialResidual,FinalResidual,StepBound
    real*8,dimension(M)::fdx
    real*8,dimension(M,N)::J

    !Initialize
    tol=[MinStepLength,Precision,1d-15,MinStepLength,MinStepLength,1d-15]
    call fd(fdx,x,M,N)
    call Jacobian(J,x,M,N)
    StepBound=100d0; RCI_request=0
    if(dtrnlsp_init(handle,N,M,x,tol,MaxIteration,MaxStepIteration,StepBound)/=TR_SUCCESS) then
        write(*,'(1x,A42)')'Trust region abort: invalid initialization'
        call mkl_free_buffers; return
    end if
    if(dtrnlsp_check(handle,N,M,J,fdx,tol,info)/=TR_SUCCESS) then
        write(*,'(1x,A32)')'Trust region abort: check failed'
        call mkl_free_buffers; return
    else
        if(info(1)/=0.or.info(2)/=0.or.info(3)/=0.or.info(4)/=0) then
            write(*,'(1x,A61)')'Trust region abort: check was not passed, the information is:'
            write(*,*)info
            call mkl_free_buffers; return
        end if
    end if
    do!Main loop
        if (dtrnlsp_solve(handle,fdx,J,RCI_request)/=TR_SUCCESS) then
            call mkl_free_buffers; return
        end if
        select case (RCI_request)
        case (-1,-2,-3,-4,-5,-6); exit
        case (1); call fd(fdx,x,M,N)
        case (2); call Jacobian(J,x,M,N)
        end select
    end do
    !Clean up
    if (dtrnlsp_get(handle,TotalIteration,StopReason,InitialResidual,FinalResidual)/=TR_SUCCESS) then
        call mkl_free_buffers; return
    end if
    if (dtrnlsp_delete(handle)/=TR_SUCCESS) then
        call mkl_free_buffers; return
    end if
    call mkl_free_buffers
    !Warn
    if(StopReason/=3.and.Warning) then
        select case(StopReason)
        case(1); write(*,'(1x,A44)')'Failed trust region: max iteration exceeded!'
        case(4); write(*,'(1x,A51)')'Failed trust region: singular Jacobian encountered!'
        case default; write(*,'(1x,A87)')'Trust region warning: step length has converged, but residual has not met accuracy goal'
        end select
        write(*,*)'Final residual =',FinalResidual
    end if
end subroutine trust_region