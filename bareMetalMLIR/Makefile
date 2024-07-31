BUILD_DIR = ${PWD}/
DIRTY_BUILD_DIR = ${BUILD_DIR}

INCLUDES = \
		   -I${BUILD_DIR}../riscv-tests \
		   -I${BUILD_DIR}../riscv-tests/env \
		   -I${BUILD_DIR}.. \
		   -I${BUILD_DIR}../riscv-tests/benchmarks/common \
		
LINKS := \
		-T${BUILD_DIR}../riscv-tests/benchmarks/common/test.ld

SRCS := \
		${BUILD_DIR}../bareMetalC/tiled_matmul_ws.c

MORE_SRCS := \
		${BUILD_DIR}../riscv-tests/benchmarks/common/syscalls.c \
		${BUILD_DIR}../riscv-tests/benchmarks/common/crt.S 

MLIR_OPTS := -DMLIR

BINARY = tiled_matmul_ws-baremetal-mlir

CLANG = clang-17
	
all: ${BINARY}

${BINARY}: tiled_matmul.o
	riscv64-unknown-elf-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc  -DID_STRING= -DPRINT_TILE=0  -nostdlib -nostartfiles -static ${INCLUDES} ${LINKS} -DBAREMETAL=1 ${MLIR_OPTS} ${SRCS} tiled_matmul.o  -o ${DIRTY_BUILD_DIR}/${BINARY} ${MORE_SRCS}

run:
	spike --extension=gemmini ${BUILD_DIR}../bareMetalMLIR/${BINARY}

CFLAGS += --target=riscv64-unknown-elf
CFLAGS += -mcpu=generic-rv64
CFLAGS += -march=rv64gc
CFLAGS += -mcmodel=medany
CFLAGS += -ffast-math
CFLAGS += -fno-builtin-printf
CFLAGS += -fno-common
CFLAGS += -nostdlib
CFLAGS += -nostartfiles
CFLAGS += -O2

tiled_matmul.o : tiled_matmul.ll
	${CLANG} ${CFLAGS} -x ir -c $^ -o $@
