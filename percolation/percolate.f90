program percolate
    implicit none

    interface
        recursive function findroot(s, sites) result(r)
            integer, intent(in) :: s
            integer, pointer, intent(inout) :: sites(:)
            integer :: r
        end function findroot 
    end interface 


    integer, parameter :: wp = kind(1.d0)

    integer :: ns, l, n, m
    integer, allocatable :: bonds(:,:)

    integer :: i, j, seed_size, temp0, temp1, next
    integer, allocatable :: seed(:)
    real(wp), allocatable :: randnums(:)

    integer, pointer :: sites(:)
    integer :: s1, s2, r1, r2, rsmall, rbig, rinf

    integer(8) :: np_inf, avg_s
    real(wp), allocatable :: p_inf(:), p2_inf(:), sm(:)

    open(10, file="bonds.txt", status="old", access="sequential", form="formatted", action="read")
    read(10,*) ns, l, n, m
    allocate(bonds(0:2,0:m-1))
    do i = 0, m-1
        read(10,*) bonds(0,i), bonds(1,i)
    end do

    call random_seed(size=seed_size)
    allocate(seed(seed_size))
    seed = 100
    call random_seed(put=seed)

    allocate(randnums(m-1))
    call random_number(randnums)
    do i = 1, m-1
        j = i-1
        next = i + int(randnums(i)*(m-i))
        temp0 = bonds(0,j)
        temp1 = bonds(1,j)
        bonds(0,j) = bonds(0, next)
        bonds(0,j) = bonds(1, next)
    end do

    allocate(p_inf(0:m), p2_inf(0:m), sm(0:m))
    p_inf(0) = 0.0_wp
    p_inf(m) = 1.0_wp
    p2_inf(0) = 0.0_wp
    p2_inf(m) = 1.0_wp 
    sm(0) = 1.0_wp
    sm(m) = 0.0_wp

    np_inf =  0
    avg_s = n
    
    allocate(sites(0:n-1))
    sites = -1

    do i = 0, m-1
        s1 = bonds(0, i)
        s2 = bonds(1, i)
        r1 = findroot(s1, sites)
        r2 = findroot(s2, sites)
        if (r1 /= r2) then
            avg_s = avg_s + 2 * int(sites(r1), 8) * int(sites(r2), 8)
            if (sites(r1) < sites(r2)) then
                rbig = r1
                rsmall = r2
            else
                rbig = r2
                rsmall = r1
            end if
            sites(rbig) = sites(rbig) + sites(rsmall)
            sites(rsmall) = rbig
            if (sites(rbig) < sites(rinf)) then
                rinf = rbig
            end if
        end if
        np_inf = -sites(rinf)
        p_inf(i+1) = real(np_inf, wp) / real(n, wp)
        p2_inf(i+1) = p_inf(i+1)**2
        sm(i+1) = real(avg_s - np_inf**2, wp) / real(n - np_inf, wp) 
    end do 
   
    open(11, file="measurements.txt", status="replace", form="formatted", access="sequential", action="write")
    write(11,*) ns, l, n, m
    do i = 0, m
        write(11,*) p_inf(i), p2_inf(i), sm(i)
    end do

    deallocate(bonds, seed, randnums, sites, p_inf, sm)

end program percolate


recursive function findroot(s, sites) result(r)
    integer, intent(in) :: s
    integer, pointer, intent(inout) :: sites(:)
    integer :: r
    if (sites(s) < 0) then
        r = s
    else
        r = findroot(sites(s), sites)
        sites(s) = r
    end if
end function findroot
