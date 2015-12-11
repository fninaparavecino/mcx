#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2009/04/02
#################################################################

BACKEND ?= cuda

ifeq ($(BACKEND),ocelot)
  CC?=g++
endif

CUDACC=/usr/local/cuda-7.0/bin/nvcc
AR=$(CC)

BINARY=mcx
OUTPUT_DIR=../bin

INCLUDEDIRS=-I/usr/local/cuda-7.0/include

ifeq ($(BACKEND),ocelot)
  LINKOPT=-L/usr/local/lib `OcelotConfig -l`
  CUCCOPT=-g -arch=sm_35 #--maxrregcount 32
else
  LINKOPT=-L/usr/local/cuda-7.0/lib -lcudart -lm
  CUCCOPT=-lineinfo #-g #-arch compute_20 #--maxrregcount 32
endif

DLLFLAG=-fPIC

CPPOPT=-g -Wall -O3 -std=c99 # -DUSE_OS_TIMER

OBJSUFFIX=.o
EXESUFFIX=

FILES=mcx_core mcx_utils mcx_shapes tictoc mcextreme cjson/cJSON

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)
ifeq ($(findstring MINGW32,$(PLATFORM)), MINGW32)
  CC=nvcc
  LINKOPT=-L/c/CUDA/lib -lcudart --link
  INCLUDEDIRS +=-I/c/CUDA/include
  CPPOPT =-c -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
  DLLFLAG=
else
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     CPPOPT +=-m64
     CUCCOPT +=-m64
     ifeq "$(wildcard /usr/local/cuda/lib64)" "/usr/local/cuda/lib64"
        ifeq ($(BACKEND),cuda)
           LINKOPT=-L/usr/local/cuda-7.0/lib64 -lcudart -lm -lstdc++
        endif
     endif
  endif
endif

CUCCOPT+=-Xcompiler -fopenmp
CPPOPT+=-fopenmp

all logfast:CUCCOPT+=-use_fast_math
mt:         CUCCOPT+=-DUSE_MT_RAND
fast:       CUCCOPT+=-DUSE_MT_RAND -use_fast_math
log:        CUCCOPT+=
debugmt:    mt
debuglog:   CUCCOPT+=
racing:     CUCCOPT+=-DTEST_RACING
mtatomic:   CUCCOPT+=-DUSE_MT_RAND -DUSE_ATOMIC -use_fast_math -arch=sm_35 -DMCX_TARGET_NAME='"MT Atomic MCX"'
logatomic:  CUCCOPT+=-DUSE_ATOMIC -use_fast_math -arch=sm_35 -DMCX_TARGET_NAME='"LL5 Atomic MCX"'
fermi fermimex fermioct:      CUCCOPT+=-DUSE_ATOMIC -use_fast_math
#fermimex:   CUCCOPT+=-DUSE_MT_RAND -use_fast_math
mtbox logbox:		CUCCOPT+=-DUSE_CACHEBOX -use_fast_math -arch=sm_35 -DMCX_TARGET_NAME='"Cachebox MCX"'
debugmt debuglog:	CUCCOPT+=-DMCX_DEBUG
mtatomic logatomic:	BINARY:=$(BINARY)_atomic
det mex oct:    CUCCOPT+=-DSAVE_DETECTORS -use_fast_math -arch=sm_35 -DMCX_TARGET_NAME='"Detective MCX"'
detbox mexbox octbox fermi fermimex fermioct:     CUCCOPT+=-DSAVE_DETECTORS -DUSE_CACHEBOX -use_fast_math -arch=sm_20
detbox mexbox octbox: CUCCOPT+=-DMCX_TARGET_NAME='"Cached Detective MCX"'
fermi fermimex fermioct: CUCCOPT+=-DMCX_TARGET_NAME='"Fermi MCX"'
det detbox: BINARY:=$(BINARY)_det
logbox detbox:          BINARY:=$(BINARY)_cached
all mt fast log logfast racing mtatomic logatomic mtbox logbox debugmt debuglog \
 det detbox fermi:      LINKOPT+=-fopenmp

mexbox mex fermimex:        AR=mex
mexbox mex fermimex:        LINKOPT+= CXXFLAGS='$$CXXFLAGS -DSAVE_DETECTORS -DUSE_CACHEBOX -DMCX_CONTAINER -fopenmp' -lgomp LDFLAGS='$$LDFLAGS -fopenmp'
mexbox mex oct octbox fermimex fermioct:    OUTPUT_DIR=../mcxlab
mex fermimex:      BINARY=mcxlab
oct fermioct:      BINARY=mcxlab.mex
mexbox:                     BINARY=mcxlab_atom
mexbox mex oct octbox fermimex fermioct:    CUCCOPT+=--compiler-options "$(DLLFLAG)" -DMCX_CONTAINER
mexbox mex oct octbox fermimex fermioct:    CPPOPT+=$(DLLFLAG) -DMCX_CONTAINER
mexbox mex fermimex:        LINKOPT+=mcxlab.cpp -cxx -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

octbox oct fermioct:        AR= CXXFLAGS='-DSAVE_DETECTORS -DUSE_CACHEBOX -fopenmp' LDFLAGS='-fopenmp' mkoctfile
oct:               BINARY=mcxlab.mex
octbox:            BINARY=mcxlab_atom.mex
octbox oct fermioct:        LINKOPT+=--mex mcxlab.cpp $(INCLUDEDIRS)

debug: CUCCOPT+=-DMCX_DEBUG
debug: fermi

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))

all mt fast log logfast racing mtatomic logatomic mtbox logbox debugmt debuglog \
 det detbox fermi mex oct mexbox octbox fermimex fermioct: cudasdk $(OUTPUT_DIR)/$(BINARY)

$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	$(AR) $(OBJS) -o $(OUTPUT_DIR)/$(BINARY) $(LINKOPT)

%$(OBJSUFFIX): %.c
	$(CC) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cu
	$(CUDACC) -c $(CUCCOPT) -o $@  $<

sassi_cu:
	/usr/local/sassi7/bin/nvcc -c -gencode arch=compute_35,code=sm_35 -lineinfo -m64 -Xcompiler -fopenmp -DUSE_ATOMIC -use_fast_math -use_fast_math -Xptxas --sassi-inst-before="memory" -Xptxas --sassi-before-args="mem-info" -g -O3 -dc -o mcx_core.o  mcx_core.cu
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -m64 -Xcompiler -fopenmp -c -o mcx_utils.o  mcx_utils.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -m64 -Xcompiler -fopenmp -c -o mcx_shapes.o  mcx_shapes.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -m64 -Xcompiler -fopenmp -c -o tictoc.o  tictoc.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -m64 -Xcompiler -fopenmp -c -o mcextreme.o  mcextreme.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -m64 -Xcompiler -fopenmp -c -o cjson/cJSON.o  cjson/cJSON.c
	/usr/local/sassi7/bin/nvcc mcx_core.o mcx_utils.o mcx_shapes.o tictoc.o mcextreme.o cjson/cJSON.o -o ../bin/mcx -gencode arch=compute_35,code=sm_35 -L/usr/local/sassi7/lib64 -lcudart -lm -lstdc++ -m64 -Xcompiler -fopenmp -L~/Desktop/SASSI/instlibs/lib -L/usr/local/sassi7/extras/CUPTI/lib64 -lcupti -lcudadevrt

sassi_nodc:
	/usr/local/sassi7/bin/nvcc -c -gencode arch=compute_35,code=sm_35 -Xptxas --sassi-inst-before="memory" -Xptxas --sassi-before-args="mem-info" -g -O3 -dc -o mcx_core.o  mcx_core.cu
	/usr/local/sassi7/bin/nvcc -I~/Desktop/SASSI/example/inc -c -gencode arch=compute_35,code=sm_35 -g -O3 -dlink -o mcx_core_dlink.o  mcx_core.o
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o mcx_utils.o  mcx_utils.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o mcx_shapes.o  mcx_shapes.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o tictoc.o  tictoc.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o mcextreme.o  mcextreme.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o cjson/cJSON.o  cjson/cJSON.c
	/usr/local/sassi7/bin/nvcc mcx_core.o mcx_core_dlink.o mcx_utils.o mcx_shapes.o tictoc.o mcextreme.o cjson/cJSON.o -o ../bin/mcx -gencode arch=compute_35,code=sm_35 -lm -lstdc++ -m64 -L/home/fanny/Desktop/SASSI/instlibs/lib -lmemdiverge -L/usr/local/sassi7/extras/CUPTI/lib64 -lcupti -lcudadevrt

sassi:
	/usr/local/sassi7/bin/nvcc -c -gencode arch=compute_35,code=sm_35 -Xptxas --sassi-inst-before="all" -Xptxas --sassi-before-args="reg-info" -g -O3 -dc -o mcx_core.o  mcx_core.cu
	/usr/local/sassi7/bin/nvcc -c -gencode arch=compute_35,code=sm_35 -Xptxas --sassi-inst-before="all" -Xptxas --sassi-before-args="reg-info" -g -O3 -dlink -o mcx_core_dlink.o  mcx_core.o
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o mcx_utils.o  mcx_utils.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o mcx_shapes.o  mcx_shapes.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o tictoc.o  tictoc.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o mcextreme.o  mcextreme.c
	/usr/local/sassi7/bin/nvcc -I/usr/local/sassi7/include -g -O3  -c -o cjson/cJSON.o  cjson/cJSON.c
	/usr/local/sassi7/bin/nvcc mcx_core.o mcx_core_dlink.o mcx_utils.o mcx_shapes.o tictoc.o mcextreme.o cjson/cJSON.o -o ../bin/mcx -gencode arch=compute_35,code=sm_35 -lm -lstdc++ -m64 -L/home/fanny/Desktop/SASSI/instlibs/lib -lopcode -L/usr/local/sassi7/extras/CUPTI/lib64 -lcupti -lcudadevrt



clean:

clean:
	-rm -f $(OBJS) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_atomic$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_det$(EXESUFFIX)
cudasdk:
	@if [ -z `which ${CUDACC}` ]; then \
	   echo "Please first install CUDA SDK and add the path to nvcc to your PATH environment variable."; exit 1;\
	fi

.DEFAULT_GOAL := fermi

