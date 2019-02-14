FC = gfortran

FCFLAGS = -g -std=f2008

FCDEBUGFLAGS = -Wall -Wextra -Warray-temporaries -Wconversion -fimplicit-none -fbacktrace -ffree-line-length-0 -fcheck=all -ffpe-trap=zero,overflow,underflow -finit-real=nan

FCOPTFLAGS =  -O3 -march=native


.PHONY: debug, optim, clean

debug: FCFLAGS += $(FCDEBUGFLAGS)
debug: nodecreate.o percolate.o

optim: FCFLAGS += $(FCOPTFLAGS)
optim: nodecreate.o percolate.o

clean:
	rm -f nodecreate.o


nodecreate.o: nodecreate.f90
	$(FC) $(FCFLAGS) nodecreate.f90 -o nodecreate.o

percolate.o: percolate.f90
	$(FC) $(FCFLAGS) percolate.f90 -o percolate.o

