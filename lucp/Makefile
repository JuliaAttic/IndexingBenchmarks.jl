#For a proper comparison, make sure to use the same BLAS library for both Julia
#and Python. For example, we can build Julia with MKL and compare against NumPy
#with MKL, such as provided by Continuum's Anaconda distribution with the mkl
#package.

JULIA=julia-intel
PYTHON=$(HOME)/anaconda3/bin/python3

.PHONY: all

all: julia.csv julia-opt.csv python.csv

julia.csv:
	cd julia && $(JULIA) benchmark.jl

julia-opt.csv:
	cd julia && $(JULIA) benchmark-opt.jl

python.csv:
	cd python && $(PYTHON) benchmark.py

python-numba.csv:
	cd python && $(PYTHON) benchmark-numba.py

