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

#define MAT_DIM_I 65
#define MAT_DIM_K 65

int main() {
const A_row_stride = MAT_DIM_I; //ele in a row of DRAM
const B_row_stride = 1; //ele in a row of DRAM
const C_row_stride = DIM;//ele in a row of DRAM

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

  size_t I_tiles = MAT_DIM_I/DIM;
  size_t K_tiles = MAT_DIM_K/DIM;

  static elem_t A[MAT_DIM_K][MAT_DIM_I] row_align(1);
  static elem_t B[MAT_DIM_K] row_align(1);
  static elem_t C[MAT_DIM_I] row_align(1);

  for (size_t i = 0; i < MAT_DIM_I; i++)
    for (size_t k = 0; k < MAT_DIM_K; k++) {
      B[k] = k; //(rand() % 32) - 16;
      A[k][i] = (i*MAT_DIM_K + k) % 8;//(rand() % 32) - 16;
  }

  sp_tiled_matvec_os(A, B, NULL, C, A_scale_factor, B_scale_factor, 0, MAT_DIM_I/DIM + (MAT_DIM_I%DIM != 0), MAT_DIM_K/DIM + (MAT_DIM_K%DIM != 0), (DIM - MAT_DIM_I%DIM)%DIM, (DIM - MAT_DIM_K%DIM)%DIM, A_row_stride, B_row_stride, 0, C_row_stride, false, false);
  gemmini_fence();

  static elem_t golden[MAT_DIM_I];
  full_t partial;
  for (size_t i = 0; i < MAT_DIM_I; i++) {
    partial = 0;
    for (size_t k = 0; k < MAT_DIM_K; k++) {  
      partial += A[k][i] * B[k];
    }
    full_t elem = partial > elem_t_max ? elem_t_max : (partial < elem_t_min ? elem_t_min : partial);
    golden[i] = elem;
  }

  printf("Check whether \"In\" and \"Out\" matrices are identical\n");
  if (!is_equal_vector(golden, C, MAT_DIM_I)) {
    printf("Input and golden matrices are different!\n");
    printf("\"golden\" matrix:\n");
    printVector(golden, MAT_DIM_I);
    printf("\"C\" matrix:\n");
    printVector(C, MAT_DIM_I);
    printf("\n");
    exit(1);

  }

  printf("Input and golden matrices are identical, as expected\n");
  printVector(C, MAT_DIM_I);
  exit(0);
}