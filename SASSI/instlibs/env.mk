export CUDA_HOME ?= /usr/local/cuda-7.0/
export SASSI_HOME ?= /usr/local/sassi7/

# Point this toward a C++-11 capable compiler (not the compiler
# itself, just its location).
export CCBIN ?= /usr/bin/

# Set this to target your specific GPU.  Note some libries use 
# CUDA features that are only supported for > compute_30.
# IMPORTANT: YOU MUST SPECIFY A REAL ARCHITECTURE.  IF YOUR
# code SETTING DOES NOT HAVE THE "sm" PREFIX, YOUR INSTRUMENTATION
# WILL NOT WORK!
export GENCODE ?= -gencode arch=compute_35,code=sm_35 \
		  -gencode arch=compute_52,code=sm_52

