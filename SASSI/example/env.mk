# Point this toward your CUDA 7 compiler.
export CUDA_HOME ?= /usr/local/cuda-7.5/
export SASSI_HOME ?= /usr/local/sassi7/

# Point this toward a C++-11 capable compiler (not the compiler
# itself, just its location).
export CCBIN ?= /usr/bin/

# Set this to target your specific GPU.  Note some libries use 
# CUDA features that are only supported for > compute_30.
# IMPORTANT: YOU MUST SPECIFY A REAL ARCHITECTURE.  IF YOUR
# code SETTING DOES NOT HAVE THE "sm" PREFIX, YOUR INSTRUMENTATION
# WILL NOT WORK!
export GENCODE ?= -gencode arch=compute_20,code=sm_20 \
		  -gencode arch=compute_35,code=sm_35 \
		  -gencode arch=compute_52,code=sm_52

# You might want to debug an instrumentation handler.  If so, 
# uncomment the line below.  Be aware that CUPTI and cuda-gdb do 
# not play nicely together, so you'll want to make sure you're
# not using any CUPTI-related libraries if you want to debug an
# instrumentation handler...
#export DEBUG = -G -g
