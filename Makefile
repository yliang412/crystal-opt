CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC = nvcc


# Check https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/ for details.
# The nvidia-smi command can also be used to check the GPU architecture.

# Uncomment the following line to use the SM75 architecture
# SM_TARGETS = $(GENCODE_SM75)
# Uncomment the following line to use the SM70 architecture
SM_TARGETS = $(GENCODE_SM70)
# Uncomment the following line to use the SM89 architecture
# SM_TARGETS = $(GENCODE_SM89)
SM_DEF     = -DSM550

GENCODE_SM75 = -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM70 = -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM89 = -gencode=arch=compute_89,code=\"sm_89,compute_89\"

NVCCFLAGS += --std=c++17 $(SM_DEF) -Xptxas="-dlcm=ca -v" -lineinfo -Xcudafe -\#

SRC = src
BIN = bin
OBJ = obj
SSB_JO_BIN = bin/ssb-jo

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
	mkdir -p bin/ssb-jo obj/ssb-jo
	for i in 11 12 13 21 22 23 31 32 33 41 42 43; do \
		mkdir -p bin/ssb-jo/q$$i; \
		mkdir -p obj/ssb-jo/q$$i; \
	done

jo_q33_naive: $(SSB_JO_BIN)/q33/0-naive $(SSB_JO_BIN)/q33/1-naive $(SSB_JO_BIN)/q33/2-naive $(SSB_JO_BIN)/q33/3-naive $(SSB_JO_BIN)/q33/4-naive $(SSB_JO_BIN)/q33/5-naive
jo_q33_lip: $(SSB_JO_BIN)/q33/0-lip $(SSB_JO_BIN)/q33/1-lip $(SSB_JO_BIN)/q33/2-lip $(SSB_JO_BIN)/q33/3-lip $(SSB_JO_BIN)/q33/4-lip $(SSB_JO_BIN)/q33/5-lip
jo_q33: jo_q33_naive jo_q33_lip

clean:
	rm -rf bin/* obj/*

submit:
	git format-patch 8181d0da8b639af18d2a221f6c5ef7e5662faf3d..HEAD --stdout > lip-gpu.patch