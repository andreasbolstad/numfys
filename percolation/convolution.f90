program convolution
    implicit none
    
    integer, parameter :: wp = kind(1.d0)

    integer, parameter :: nq = 1000

    integer :: i, j, ns, l, n, m
    real(wp), allocatable :: lncr(:)
    
    real(wp), allocatable :: p_inf(:), p2_inf(:), sm(:)
    
    real(wp) :: logcoeff, coeff, q, pq, p2q, smq, chiq

    character(len=32) :: lstr
    character(len=100) :: filename

    
    open(10, file="measurements.txt", status="old", access="sequential", form="formatted", action="read")
    read(10,*) ns, l, n, m

    allocate(p_inf(0:m), p2_inf(0:m), sm(0:m))
    do i = 0, m
       read(10,*) p_inf(i), p2_inf(i), sm(i)
    end do
    
    close(10)
    
    
    allocate(lncr(0:m))
    lncr(0) = 0
    lncr(m) = 0
    
    do i = 1, m-1
        lncr(i) = lncr(i-1) + log(real(m-i+1,wp)) - log(real(i,wp))
    end do

    write(lstr,'(I0)') l
    filename = "data/" // trim(lstr) // "x" // trim(lstr) // "convoluted.txt"
    open(11, file=filename, status="replace", access="sequential", form="formatted", action="write")
    write(11,*) ns, l, n, m   
    
    do i = 1, nq-1
        q = real(i,wp) / real(nq, wp)
        pq = 0.0_wp
        p2q = 0.0_wp
        smq = 0.0_wp
        do j = 1, m-1
            logcoeff = lncr(j) + j*log(q) + (m-j)*log(1-q)
            if (logcoeff < -100.0_wp) then
                coeff = 0.0000001_wp
            else
                coeff = exp(logcoeff)
            end if
            pq = pq + coeff * p_inf(j)
            p2q = p2q + coeff * p2_inf(j)
            smq = smq + coeff * sm(j)
        end do
        chiq = n*sqrt(p2q - pq**2)
        write(11, *) pq, smq, chiq 
    end do

    close(11)

    deallocate(lncr)
    deallocate(p_inf, p2_inf, sm)


end program convolution

