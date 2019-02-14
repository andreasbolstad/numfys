program percolate
    implicit none

    integer, parameter :: wp = kind(1.d0)

    integer :: ns, l, n, m
    integer, allocatable :: bonds(:,:)

    integer :: i, j, seed_size, temp0, temp1, next
    integer, allocatable :: seed(:)
    real(wp), allocatable :: randnums(:)
    open(10, file="bonds.txt", status="old", access="sequential", form="formatted", action="read")
    read(10,*) ns, l, n, m
    allocate(bonds(0:2,0:m-1))
    do i = 0, m-1
        read(10,*) bonds(0,i), bonds(1,i)
    end do

    print*, ns, l, n, m
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



end program percolate
