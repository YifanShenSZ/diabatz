# intel and gnu compilers are supported
compiler = intel
flag = -O3

# User does not have to take care of following variables
gnumkl = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl
src = $(addprefix source/, trust_region.f90)

libFopt.a: $(src)
ifeq ($(compiler),intel)
	ifort -fpp -parallel -mkl -static-intel -ipo $(flag) -c $^
	xiar rcs libFopt.a *.o
	rm *.mod *.o
else
	gfortran -cpp -ffree-line-length-0 -fno-range-check -I${MKLROOT}/include $(flag) -c $^
	ar rcs libFopt.a *.o
	rm *.mod *.o
endif

.PHONY: install
install: | lib
	mv *.a lib

lib:
	mkdir lib
