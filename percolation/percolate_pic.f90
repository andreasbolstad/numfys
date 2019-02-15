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

    integer, allocatable :: clustersites(:)
    integer :: capture_pcts(6) = (/30, 40, 48, 49, 50, 70/)
    integer :: capture_nsites(6)

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
    allocate(sites(0:n-1))

    call random_number(randnums)
    do i = 1, m-1
        j = i-1
        next = i + int(randnums(i)*(m-i))
        temp0 = bonds(0,j)
        temp1 = bonds(1,j)
        bonds(0,j) = bonds(0, next)
        bonds(0,j) = bonds(1, next)
    end do

    sites = -1

    open(11, file="clusters.txt", status="replace", form="formatted", access="sequential", action="write")
    write(11,*) ns, l, n, m
    write(11,*) capture_pcts
    
    capture_nsites = capture_pcts * n / 100

    allocate(clustersites(0:n-1))

    do i = 0, m-1
        s1 = bonds(0, i)
        s2 = bonds(1, i)
        r1 = findroot(s1, sites)
        r2 = findroot(s2, sites)
        if (r1 /= r2) then
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
        if (any(capture_nsites == i)) then
            print*, "Here"
            clustersites = 0
            do j = 0, n-1
                if (findroot(j,sites) == rinf) then
                    clustersites(j) = 1
                end if
            end do
            write(11,*) clustersites
        end if 
    end do

    deallocate(bonds, seed, clustersites)

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


