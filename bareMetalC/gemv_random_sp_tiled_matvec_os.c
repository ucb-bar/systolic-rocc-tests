// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#define A_scale_factor 1.0
#define B_scale_factor 1.0
// #define C_scale_factor 1.0
#define scale 1.0
#define stride

int main() {
const A_row_stride = DIM; //ele in a row of DRAM
const B_row_stride = 1; //ele in a row of DRAM
const C_row_stride = DIM;//ele in a row of DRAM
const A_stride = 1; //ele in a row of SPAD
// const B_stride = 1; //ele in a row of SPAD
const C_stride = 1;//ele in a row of SPAD
const sizeof_C = sizeof(elem_t);

// const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
gemmini_extended3_gemv_config_ex(OUTPUT_STATIONARY, 0 & 3, 0, 1, C_stride, A_stride, false, false, false);
gemmini_extended_config_st(C_row_stride * sizeof_C, 0 & 3, scale);
// gemmini_extended3_config_ld(DIM * A_row_stride * sizeof(elem_t), A_scale_factor, false, 0);
// gemmini_extended3_config_ld(1 * sizeof(elem_t), B_scale_factor, false, 1);

#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
  printf("Flush Gemmini TLB of stale virtual addresses\n");
  gemmini_flush(0);
  printf("Initialize our input and golden matrices in main memory\n");

  //YIKES!!!

  size_t I_tiles = 2;
  size_t K_tiles = 2;

  size_t I = I_tiles*DIM;
  size_t K = K_tiles*DIM;

  elem_t B[K];
  elem_t C[I];
  elem_t A[K][I];
  full_t golden[I];
  for (size_t i = 0; i < I; i++)
    for (size_t k = 0; k < K; k++) {
      B[k] = (rand() % 64) - 32;
      A[k][i] = (rand() % 64) - 32;
  }

  sp_tiled_matvec_os(A, B, NULL, C, A_scale_factor, B_scale_factor, 0, I_tiles, K_tiles, 0, 0, A_row_stride, B_row_stride, 0, C_row_stride, false, false);
  gemmini_fence();

  for (size_t i = 0; i < I; i++) {
    golden[i] = 0;
    for (size_t k = 0; k < K; k++) {  
      golden[i] += A[k][i] * B[k];
    }
    golden[i] = golden[i] > elem_t_max ? elem_t_max : golden[i];
    golden[i] = golden[i] < elem_t_min ? elem_t_min : golden[i];
  }

  printf("Check whether \"In\" and \"Out\" matrices are identical\n");
  if (!is_equal_vector(B, C, I)) {
    printf("Input and golden matrices are different!\n");
    printf("\"golden\" matrix:\n");
    printVector(golden, I);
    printf("\"C\" matrix:\n");
    printVector(C, I);
    printf("\n");
    exit(1);

  }

  printf("Input and golden matrices are identical, as expected\n");
  printVector(C, I);
  exit(0);
}