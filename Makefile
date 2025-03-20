CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC = nvcc


# Check https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/ for details.
# The nvidia-smi command can also be used to check the GPU architecture.
SM_TARGETS = -gencode=arch=compute_75,code=\"sm_75,compute_75\"
SM_DEF     = -DSM550

NVCCFLAGS += --std=c++11 $(SM_DEF) -Xptxas="-dlcm=ca -v" -lineinfo -Xcudafe -\#

SRC = src
BIN = bin
OBJ = obj

CUB_DIR = cub/

INCLUDES = -I$(CUB_DIR) -I$(CUB_DIR)test -I. -I$(INC)

$(OBJ)/%.o: $(SRC)/%.cu
	$(NVCC) -lcurand $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(BIN)/%: $(OBJ)/%.o
	$(NVCC) $(SM_TARGETS) -lcurand $^ -o $@

setup:
	if [ ! -d "cub"  ]; then \
    wget https://github.com/NVlabs/cub/archive/1.6.4.zip; \
    unzip 1.6.4.zip; \
    mv cub-1.6.4 cub; \
    rm 1.6.4.zip; \
	fi
	mkdir -p bin/ssb obj/ssb
	mkdir -p bin/ops obj/ops

clean:
	rm -rf bin/* obj/*
