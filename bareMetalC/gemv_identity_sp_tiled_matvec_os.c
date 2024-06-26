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

  size_t I = DIM*DIM;
  size_t K = DIM*DIM;

  elem_t B[K];
  elem_t C[I];
  elem_t A[K][I];

  for (size_t i = 0; i < I; i++)
    for (size_t k = 0; k < K; k++) {
      B[k] = k + 1;
      A[k][i] = i*K +k;
  }

  sp_tiled_matvec_os(A, B, NULL, C, A_scale_factor, B_scale_factor, 0, I/DIM, K/DIM, 0, 0, A_row_stride, B_row_stride, 0, C_row_stride, false, false);
  gemmini_fence();

  printf("Check whether \"In\" and \"Out\" matrices are identical\n");
  if (!is_equal_vector(B, C, I)) {
    printf("Input and output matrices are not identical!\n");
    printf("\"B\" matrix:\n");
    printVector(B, I);
    printf("\"C\" matrix:\n");
    printVector(C, I);
    printf("\n");
    exit(1);

  }

  printf("Input and output matrices are identical, as expected\n");
  printVector(C, I);
  exit(0);
}


