BUILD_DIR = ${PWD}/
DIRTY_BUILD_DIR = ${BUILD_DIR}

INCLUDES = \
		   -I${BUILD_DIR}../riscv-tests \
		   -I${BUILD_DIR}../riscv-tests/env \
		   -I${BUILD_DIR}.. \
		   -I${BUILD_DIR}../riscv-tests/benchmarks/common \
		
LINKS := \
		-T${BUILD_DIR}../riscv-tests/benchmarks/common/test.ld

SRC := \
		${BUILD_DIR}../bareMetalC/tiled_matmul_ws.c

MORE_SRCS := \
		${BUILD_DIR}../riscv-tests/benchmarks/common/syscalls.c \
		${BUILD_DIR}../riscv-tests/benchmarks/common/crt.S 

MLIR_OPTS := -DMLIR

BINARY = tiled_matmul_128x128.x 

BINARY_ACCFG = \
			   tiled_matmul_512x512.acc_dialect.x \
			   tiled_matmul_256x256.acc_dialect.x \
			   tiled_matmul_128x128.acc_dialect.x \
			   tiled_matmul_64x64.acc_dialect.x \
			   tiled_matmul_32x32.acc_dialect.x

BINARY_NO_OPT_ACCFG = \
			   tiled_matmul_512x512.no_opt.acc_dialect.x \
			   tiled_matmul_256x256.no_opt.acc_dialect.x \
			   tiled_matmul_128x128.no_opt.acc_dialect.x \
			   tiled_matmul_64x64.no_opt.acc_dialect.x \
			   tiled_matmul_32x32.no_opt.acc_dialect.x

CLANG = clang-17
MLIR_OPT := mlir-opt-17
MLIR_TRANSLATE := mlir-translate-17
SNAX_OPT := snax-opt

TILE_SIZE = 128
	
all:
	$(info RUN_ACCFG = $(RUN_ACCFG))

# Default rule, for 128x128
tiled_matmul_128x128.x: tiled_matmul_128x128.o 
	riscv64-unknown-elf-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc  -DID_STRING= -DPRINT_TILE=0  -nostdlib -nostartfiles -static ${INCLUDES} ${LINKS} -DBAREMETAL=1 ${MLIR_OPTS} ${SRC} $<  -o ${DIRTY_BUILD_DIR}/$@ ${MORE_SRCS}


# Use special magic to select right C file for MLIR accfg experiments with different sizes
tiled_matmul_%.acc_dialect.x: tiled_matmul_%.acc_dialect.o ${BUILD_DIR}../bareMetalC/tiled_matmul_ws_%.c
	riscv64-unknown-elf-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc  -DID_STRING= -DPRINT_TILE=0  -nostdlib -nostartfiles -static ${INCLUDES} ${LINKS} -DBAREMETAL=1 ${MLIR_OPTS} $(word 2, $^) $<  -o ${DIRTY_BUILD_DIR}/$@ ${MORE_SRCS}


# Use special magic to select right C file for MLIR accfg experiments with different sizes
tiled_matmul_%.no_opt.acc_dialect.x: tiled_matmul_%.acc_dialect.o ${BUILD_DIR}../bareMetalC/tiled_matmul_ws_%.c
	riscv64-unknown-elf-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc  -DID_STRING= -DPRINT_TILE=0  -nostdlib -nostartfiles -static ${INCLUDES} ${LINKS} -DBAREMETAL=1 ${MLIR_OPTS} $(word 2, $^) $<  -o ${DIRTY_BUILD_DIR}/$@ ${MORE_SRCS}

sim_%: %
	spike --extension=gemmini ${BUILD_DIR}../bareMetalMLIR/$<

RUN_ACCFG = $(addprefix run_, $(BINARY_ACCFG))
run-accfg: $(RUN_ACCFG)
$(RUN_ACCFG): run_%: sim_%

RUN = $(addprefix run_, $(BINARY))
run: $(RUN)
$(RUN): run_%: sim_%

all_binaries: $(BINARY_ACCFG)

all_binaries_no_opt: $(BINARY_NO_OPT_ACCFG)


%.mlir: %.transform.mlir
	${MLIR_OPT} --pass-pipeline='builtin.module(test-transform-dialect-interpreter{bind-first-extra-to-ops=linalg.quantized_matmul}, test-transform-dialect-erase-schedule)' -o $@ $^

%.acc_dialect.mlir: %.transform.mlir
	${MLIR_OPT} --mlir-print-op-generic --pass-pipeline='builtin.module(test-transform-dialect-interpreter{bind-first-extra-to-ops=linalg.quantized_matmul}, test-transform-dialect-erase-schedule, linalg-generalize-named-ops)' $^ | sed -E 's/iterator_types =/library_call="gemmini", iterator_types =/gm;t' | ${SNAX_OPT} -p insert-accfg-op{accelerator=gemmini},convert-linalg-to-accfg,mlir-opt{executable=mlir-opt-17\ generic=true\ arguments=-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic,-split-input-file},accfg-dedup,convert-accfg-to-csr -o $@

%.no_opt.acc_dialect.mlir: %.transform.mlir
	${MLIR_OPT} --mlir-print-op-generic --pass-pipeline='builtin.module(test-transform-dialect-interpreter{bind-first-extra-to-ops=linalg.quantized_matmul}, test-transform-dialect-erase-schedule, linalg-generalize-named-ops)' $^ | sed -E 's/iterator_types =/library_call="gemmini", iterator_types =/gm;t' | ${SNAX_OPT} -p insert-accfg-op{accelerator=gemmini},convert-linalg-to-accfg,convert-accfg-to-csr -o $@

# replace linalg quantized matmul with function call
replace_string = 's/linalg\.quantized_matmul ins\(%subview, %subview_7, %c0_i32, %c0_i32 : memref<\?x128xi8, strided<\[128, 1\], offset: \?>>, memref<128x\?xi8, strided<\[128, 1\], offset: \?>>, i32, i32\) outs\(%subview_8 : memref<\?x\?xi32, strided<\[128, 1\], offset: \?>>\)/func\.call @sp_tiled_matmul_ws(%subview,%subview_7,%subview_8) : (memref<?x${TILE_SIZE}xi8, strided<\[${TILE_SIZE}, 1\], offset: \?>>, memref<${TILE_SIZE}x?xi8, strided<\[${TILE_SIZE}, 1\], offset: \?>>, memref<?x?xi32, strided<\[${TILE_SIZE}, 1\], offset: \?>>) -> ()/gm;t'

tiled_matmul_128x128.func.mlir: tiled_matmul_128x128.mlir
	sed -E ${replace_string} < $^ > $@

replace_string_2 = 's/  }\n}/}\nfunc.func private @sp_tiled_matmul_ws(%subview: memref<?x${TILE_SIZE}xi8, strided<\[${TILE_SIZE}, 1\], offset: ?>>,%subview_7: memref<${TILE_SIZE}x?xi8, strided<\[${TILE_SIZE}, 1\], offset: \?>>, %subview_8:memref<?x?xi32, strided<\[${TILE_SIZE}, 1\], offset: \?>>)\n}/gm;t'

# Add function declaration
tiled_matmul_128x128.func2.mlir: tiled_matmul_128x128.func.mlir
	sed -zE ${replace_string_2} < $^ > $@


MLIROPTFLAGS =
MLIROPTFLAGS += --convert-linalg-to-loops
MLIROPTFLAGS += --convert-scf-to-cf
MLIROPTFLAGS += --lower-affine
MLIROPTFLAGS += --canonicalize
MLIROPTFLAGS += --cse
MLIROPTFLAGS += --convert-math-to-llvm
MLIROPTFLAGS += --llvm-request-c-wrappers
MLIROPTFLAGS += --expand-strided-metadata
MLIROPTFLAGS += --lower-affine
MLIROPTFLAGS += --convert-index-to-llvm=index-bitwidth=64
MLIROPTFLAGS += --convert-cf-to-llvm=index-bitwidth=64
MLIROPTFLAGS += --convert-arith-to-llvm=index-bitwidth=64
MLIROPTFLAGS += --convert-func-to-llvm='index-bitwidth=64'
MLIROPTFLAGS += --finalize-memref-to-llvm='use-generic-functions index-bitwidth=64'
MLIROPTFLAGS += --canonicalize
MLIROPTFLAGS += --reconcile-unrealized-casts


%.ll.mlir : %.func2.mlir
	${MLIR_OPT} ${MLIROPTFLAGS} -o $@ $^

%.acc_dialect.ll.mlir : %.acc_dialect.mlir
	${MLIR_OPT} ${MLIROPTFLAGS} -o $@ $^

%.ll : %.ll.mlir
	${MLIR_TRANSLATE} --mlir-to-llvmir -o $@ $^

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

%.o : %.ll
	${CLANG} ${CFLAGS} -x ir -c $^ -o $@
