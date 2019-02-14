subroutine randwalk(times, nw, max_t)
        integer, parameter :: seed = 84982
        integer, intent(in) :: nw, max_t
        integer, intent(out), dimension(max_t) :: times
        integer :: i, iw, t
        real :: x, y
        integer :: is_positive
        real :: start, finish

        call srand(seed)
        times = 0
        call cpu_time(start)
        do iw = 1, nw
                call random_number(x)
                if (x > 0) then
                        is_positive = 1
                else
                        is_positive = -1
                end if
                do t = 1, max_t
                        call random_number(y)
                        x = x + y * 2 - 1
                        if (x * is_positive < 0) then
                                times(t) = times(t) + 1
                                exit
                        end if
                end do
        end do
        call cpu_time(finish)
        print *, "Time (seconds): ", finish-start
end subroutine randwalk
