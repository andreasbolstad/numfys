program nodecreate
    implicit none


    integer :: i, j, l, n, m, ns
    character(len=32) :: arg1, arg2

    integer, allocatable :: bonds(:,:)


    ! Get data from commandline arguments
    call get_command_argument(1, arg1)
    read(arg1,*) l

    if (mod(l,2) /= 0) then
        write(*,*) "Error, need even number of sites"
        stop
    end if
    
    call get_command_argument(2, arg2)
    read(arg2,*) ns  ! Neighbouring sites
    
    if (.not. any((/4, 6, 3/) == l)) then
        write(*,*) "Error, number of bonds per site must be 3, 4 or 6"
        stop
    end if


    ! Start of main program

    n = l*l ! Total number of sites
    m = ns*n/2 ! Number of bonds
    
    allocate(bonds(0:1,0:m-1))
    select case (ns)
        case (4) ! Square
            do i = 0, n-1
                j = i*2
                ! Right
                bonds(0,j) = i
                bonds(1,j) = i/l*l + mod(i+1, l)
                ! Down
                bonds(0,j+1) = i
                bonds(1,j+1) = mod(i+l,n)
            end do

        case (6) ! Triangular
            do i = 0, n-1
                j = i*3
                ! Right
                bonds(0,j) = i
                bonds(1,j) = i/l*l + mod(i+1, l)
                ! Down
                bonds(0,j+1) = i
                bonds(1,j+1) = mod(i+l,n)
                ! Right-down
                bonds(0, j+2) = i
                bonds(1, j+2) = mod(i/l*l + mod(i+1,l) + l, n)
            end do

        case (3) ! Honeycomb
            j = 0
            do i = 0, n-1
                if (mod(i,2) == mod(i/l,2)) then
                    bonds(0,j) = i
                    bonds(1,j) = i/l*l + mod(i+1,l)
                    j = j + 1
                end if
                bonds(0,j) = i
                bonds(1,j) = mod(i+l, n)
                j = j + 1
            end do
    end select
   
    do i = 0, m-1
        print*, bonds(0,i), bonds(1,i)
    end do

    deallocate(bonds)

end program nodecreate
