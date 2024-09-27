// See LICENSE for license details.
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#define ReRoCC 1
#include "include/gemmini_testutils.h"

#define ACTIVATION NO_ACTIVATION

#define NO_BIAS 1
#define REPEATING_BIAS 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#ifndef BAREMETAL

#define MAT_DIM_I 128
#define MAT_DIM_K 512
#define MAT_DIM_J 256

#else

#define MAT_DIM_I 64
#define MAT_DIM_K 64
#define MAT_DIM_J 64

#endif

#if A_TRANSPOSE==0
#define A_STRIDE MAT_DIM_K
#else
#define A_STRIDE MAT_DIM_I
#endif

#if B_TRANSPOSE==0
#define B_STRIDE MAT_DIM_J
#else
#define B_STRIDE MAT_DIM_K
#endif

#define SYNC_SIZE 0
#define NUM_ARRAY 2
#define BASE_ADDR 0x70000000L
#define ADDR_OFFSET 0x100000L

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

#if ReRoCC == 1
    for(int i = 0; i < NUM_ARRAY; i++){
        bool acquired = false;
        while(!acquired){
            acquired = rr_acquire_single(i, i);
        }
    }
    for(int i = 0; i < NUM_ARRAY; i++){
        rr_set_opc(XCUSTOM_ACC, i);
        gemmini_flush(0);
    }
#endif

#if A_TRANSPOSE==0
    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
#else
    static elem_t full_A[MAT_DIM_K][MAT_DIM_I] row_align(MAX_BLOCK_LEN);
#endif

#if B_TRANSPOSE==0
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
#else
    static elem_t full_B[MAT_DIM_J][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
#endif

    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
    static acc_t full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC);

    //static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];
    int8_t tile_I = MAT_DIM_I / DIM;
    int8_t tile_J = MAT_DIM_J / DIM;
    int8_t tile_K = MAT_DIM_K / DIM;

#if ReRoCC == 1
    rr_set_opc(XCUSTOM_ACC, 0);
#endif

    elem_t* temp_addr = (elem_t*)BASE_ADDR + ADDR_OFFSET;
    //memcpy((elem_t*) temp_addr, (elem_t*) full_A, sizeof(elem_t)*MAT_DIM_I*MAT_DIM_K);


    printf("temp addr: 0x%08lx\n", temp_addr);
    //printf("Starting gemmini matmul\n");
    printf("I: %d, J: %d, K: %d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    printf("tile_I: %d, tile_J: %d, tile_K: %d\n", tile_I, tile_J, tile_K);
    //printf("NO_BIAS: %d, REPEATING_BIAS: %d\n", NO_BIAS, REPEATING_BIAS);
    //printf("A_TRANSPOSE: %d, B_TRANSPOSE: %d\n", A_TRANSPOSE, B_TRANSPOSE);
    uint64_t start = read_cycles();   

    tiled_matmul_small(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], temp_addr,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            tile_I, tile_J, tile_K,
            ACTIVATION, ACC_SCALE_IDENTITY, REPEATING_BIAS,
            false, false,
            false, false,
            0, 0,
            SYNC_SIZE,
            true);

#if ReRoCC == 1
    rr_fence(0);
#endif

    printf("first one done\n");
#if ReRoCC == 1
    rr_set_opc(XCUSTOM_ACC, 1);
#endif

    tiled_matmul_small(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            NULL, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            tile_I, tile_J, tile_K,
            ACTIVATION, ACC_SCALE_IDENTITY, REPEATING_BIAS,
            false, false,
            false, false,
            0, 0,
            SYNC_SIZE,
            true);

#if ReRoCC == 1
    rr_fence(1);
#endif

      uint64_t end = read_cycles();
      printf("Cycles taken: %llu\n", end-start);

      const uint64_t total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
      const uint64_t ideal_cycles = total_macs / (DIM * DIM);
      const uint64_t utilization = 100 * ideal_cycles / (end-start);
      printf("Total macs: %llu\n", total_macs);
      printf("Ideal cycles: %llu\n", ideal_cycles);
      printf("Utilization: %llu%%\n", utilization);
    //printf("RDMA_BYTES_REC: %u\n", counter_read(0));
    //printf("WDMA_BYTES_SENT: %u\n", counter_read(1));
#if ReRoCC == 1
    for(int i = 0; i < NUM_ARRAY; i++){
        rr_release(i);
    }
#endif

  exit(0);
}

