// See LICENSE for license details.
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
#include "util.h"
#include "include/rerocc.h"

#define BASE_ADDR 0x70000000L
#define ADDR_OFFSET 0x100000L

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0
#define NO_BIAS 1
#define REPEATING_BIAS 1
#define NUM_ARRAY 1
#define ACTIVATION NO_ACTIVATION

#define SYNC_SIZE0 1
#define SYNC_SIZE1 0
#define REPEAT 2

#define MAT_DIM_I 128
#define MAT_DIM_K 64
#define MAT_DIM_J 128
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

int init_random(elem_t * buf, int len) {
    for (elem_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
        int rv = rand() % 100;
        int value = 0;
        if (rv < 30){
          value = rand() % 7 + 1;
        }
        *ptr = value;
    }
 
    //for (elem_t * ptr = buf; ptr < buf + len; ptr++)
    //  *ptr = rand() % 5 - 2;
    int num_zero = 0;
    for (elem_t* ptr = buf; ptr < buf + len; ptr++)
      if(*ptr == 0)
        num_zero ++;
    int sparsity = (int)((100*num_zero) / len);
    //printf("sparsity: %d\n", (int)((100 * num_zero) / len));
    return sparsity;
}

// EDIT THIS
static size_t nc = 2;

void __main(void) {
  size_t mhartid = read_csr(mhartid);

  int cid = mhartid;
  for(int i = 0; i < nc; i++){
    if (i == cid) printf("Thread %d/%d starting\n", mhartid, nc);
    barrier(nc);
  }

  if (mhartid >= nc) while (1);

  // this seems max tile 
  /*
  int tile_I = 4;
  int tile_J = 4;
  int tile_K = 16;
  int dim_I = DIM * tile_I;
  int dim_J = DIM * tile_J;
  int dim_K = DIM * tile_K;
  int tile_macs = dim_I * dim_J * dim_K;
  int num_en_array = is_float ? NUM_FP : NUM_INT;
  */
#if ReRoCC == 1
  for(int j = 0; j < nc; j++){
    if(j == cid && j == 0){
      for(int i = 0; i < NUM_ARRAY; i++){
        int cfgid = i;
        int acc_id = i;
        bool acquired = false;
        while(!acquired){
          acquired = rr_acquire_single(cfgid, acc_id);
        }
        rr_set_opc(XCUSTOM_ACC, cfgid);
        gemmini_flush(0);    
      }
    }
    else if(j == cid && j == 1){
      for(int i = 0; i < NUM_ARRAY; i++){
        int cfgid = i;
        int acc_id = i + NUM_ARRAY;
        bool acquired = false;
        while(!acquired){
          acquired = rr_acquire_single(cfgid, acc_id);
        }
        rr_set_opc(XCUSTOM_ACC, cfgid);
        gemmini_flush(0);      
      } 
    }
  }

  for(int i = 0; i < nc; i++){
    if (i == cid) printf("Thread %d acquire done\n", i);
    barrier(nc);
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

  for(int i = 0; i < nc; i++){
    if (i == cid) printf("Start matmul\n");
    barrier(nc);
  }
  uint64_t num_mac = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
  uint64_t ideal_cycles = num_mac / (DIM * DIM);

  uint64_t start, end;
  for(int r = 0; r < REPEAT; r ++){
    start = read_cycles();
    if(cid == 0){
      for(int i = 0; i < NUM_ARRAY; i++){
        tiled_matmul_sync_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
              (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
              A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
              MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
              ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
              A_TRANSPOSE, B_TRANSPOSE,
              false, false,
              0,
              SYNC_SIZE0);
      }
    }
    else {
      for(int i = 0; i < NUM_ARRAY; i++){
        tiled_matmul_sync_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
              (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
              A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
              MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
              ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
              A_TRANSPOSE, B_TRANSPOSE,
              false, false,
              0,
              SYNC_SIZE1);
      }
    }
    for (int i = 0; i < NUM_ARRAY; i++)
      rr_fence(i);
    end = read_cycles();
    barrier(nc);

    const uint64_t utilization = 100 * ideal_cycles / (end-start);
    for(int i = 0; i < nc; i++){
      if (i == cid) printf("%d Thread %d cycles: %d (%d%%)\n", r, cid, end-start, utilization);
      barrier(nc);
    }
  }

  for (int i = 0; i < NUM_ARRAY; i++)
    rr_release(i);

  for(int i = 0; i < nc; i++){
    if (i == cid) printf("Thread %d/%d finished\n", cid, nc);
    barrier(nc);
  }

  // Spin if not core 0
  if (mhartid > 0) while (1);
}


int main(void) {
  __main();
  return 0;
}