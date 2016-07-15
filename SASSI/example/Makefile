TARGET := kernelMatrixMul
OBJECTS := kernelMatrixMul.o

# Copy the environment from env.mk
include ../instlibs/env.mk

# Make sure NVCC uses the C++ compiler we want it to.
export PATH := $(CCBIN):$(PATH)

# What example do you want to build?
EXAMPLES := branch cfg none

# Point to SASSI's copy of nvcc.
NVCC := $(SASSI_HOME)/bin/nvcc

# Definitions for the insertion sites.
AFTER_REG := -Xptxas --sassi-inst-after="reg-writes" 
BEFORE_ALL := -Xptxas --sassi-inst-before="all"
BEFORE_COND_BRANCHES := -Xptxas --sassi-inst-before="cond-branches"

# Definitions for the arguments to pass to handlers.
AFTER_REG_INFO := -Xptxas --sassi-after-args="reg-info"
BEFORE_COND_BRANCH_INFO := -Xptxas --sassi-before-args="cond-branch-info"

# If we want to skip predicated off instructions, include this option.
IFF := -Xptxas --sassi-iff-true-predicate-handler-call

# If we want to debug.
DEBUG := -Xptxas --sassi-debug

# The location of our instrumentation libraries.
INST_LIB_DIR = ../instlibs/lib

# We rely heavily on CUPTI to let us know when kernels are launched.
CUPTI_LIB_DIR = $(SASSI_HOME)/extras/CUPTI/lib64
CUPTI = -L$(CUPTI_LIB_DIR) -lcupti 

LINK_FLAGS += -lcudadevrt -Xlinker -rpath,$(CUPTI_LIB_DIR)
NVCC_FLAGS += -g -O3 -dc
CXX_FLAGS += -g -O3

# If you want to "intercept" main and exit, use this GCC-specific option.
# We only use this option for Fermi examples, where we don't have the luxury
# of using UVM.
LDWRAP := -Xlinker "--wrap=main" -Xlinker "--wrap=exit"

# Depending on the experiment, let's choose different SASSI options.
branch: EXTRA_NVCC_FLAGS = $(LDWRAP) -lineinfo $(BEFORE_COND_BRANCHES) $(BEFORE_COND_BRANCH_INFO) $(NVCC_FLAGS) 
branch: EXTRA_LINK_FLAGS = $(LDWRAP) -L$(INST_LIB_DIR) -lbranch $(CUPTI) $(LINK_FLAGS) -lboost_regex -lcrypto

cfg: EXTRA_NVCC_FLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry $(NVCC_FLAGS)
cfg: EXTRA_LINK_FLAGS = $(LINK_FLAGS) -L$(INST_LIB_DIR) -lcfg $(CUPTI)

none: EXTRA_NVCC_FLAGS = $(NVCC_FLAGS)
none: EXTRA_LINK_FLAGS = $(LINK_FLAGS) 

$(EXAMPLES): $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) -o $@ $^ $(GENCODE) $(EXTRA_LINK_FLAGS)

%.o:%.cu
	$(NVCC) -I./inc  -c $(GENCODE) $(EXTRA_NVCC_FLAGS) -o $@ $^ 

%.o:%.cpp
	$(NVCC) -I./inc  -c $(CXX_FLAGS) -o $@ $^ 

run: $(TARGET)
	./$(TARGET)

clean: 
	$(RM) -f $(TARGET) *.o

