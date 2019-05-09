FC = gfortran

FCFLAGS = -g -std=f2008

FCDEBUGFLAGS = -Wall -Wextra -Warray-temporaries -Wconversion -fimplicit-none -fbacktrace -ffree-line-length-0 -fcheck=all -ffpe-trap=zero,overflow,underflow -finit-real=nan

FCOPTFLAGS =  -O3 -march=native


.PHONY: debug, optim, clean

optim: FCFLAGS += $(FCOPTFLAGS)
optim: nodecreate percolate percolate_pic convolution

debug: FCFLAGS += $(FCDEBUGFLAGS)
debug: nodecreate percolate percolate_pic convolution

clean:
	rm -f nodecreate percolate percolate_pic convolution
	rm -f bonds.txt clusters.txt measurements.txt
	rm -rf data
	rm -rf figures

nodecreate: nodecreate.f90
	$(FC) $(FCFLAGS) nodecreate.f90 -o nodecreate

percolate: percolate.f90
	$(FC) $(FCFLAGS) percolate.f90 -o percolate

percolate_pic: percolate_pic.f90
	$(FC) $(FCFLAGS) percolate_pic.f90 -o percolate_pic

convolution: convolution.f90
	mkdir -p data
	$(FC) $(FCFLAGS) convolution.f90 -o convolution


