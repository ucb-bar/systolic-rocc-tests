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
const B_row_stride = DIM; //ele in a row of DRAM
const C_row_stride = DIM;//ele in a row of DRAM
const A_stride = 1; //ele in a row of SPAD
// const B_stride = 1; //ele in a row of SPAD
const C_stride = 1;//ele in a row of SPAD
const sizeof_C = sizeof(elem_t);

// const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
gemmini_extended3_gemv_config_ex(WEIGHT_STATIONARY, 0 & 3, 0, 1, C_stride, A_stride, false, false, false);
gemmini_extended_config_st(C_row_stride * sizeof_C, 0 & 3, scale);
gemmini_extended3_config_ld(DIM * A_row_stride * sizeof(elem_t), A_scale_factor, false, 0);
gemmini_extended3_config_ld(B_row_stride * sizeof(elem_t), B_scale_factor, false, 1);
//gemmini_extended_config_ex(WS, 0 & 3, 0, 1, false, false);
// gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
// gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, ACC_SCALE_IDENTITY, A_stride, A_transpose, B_transpose, false)
// gemmini_extended_gemv_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides) \
// gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
// gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
// gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1)
// gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
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
  printf("Initialize our input and output matrices in main memory\n");
  elem_t B[DIM];
  elem_t C[DIM];
  elem_t A[DIM][DIM];
  for (size_t i = 0; i < DIM; i++) {
    B[i] = i;
    for (size_t j = 0; j < DIM; j++)
      A[i][j] = i == j;
}

  sp_tiled_matvec_ws(A, B, C, A_scale_factor, B_scale_factor, DIM,1, 1, 0, 0, 0, A_row_stride, C_row_stride, false, false);
  gemmini_fence();

  printf("Check whether \"In\" and \"Out\" matrices are identical\n");
  if (!is_equal_vector(B, C)) {
    printf("Input and output matrices are different!\n");
    printf("\"B\" matrix:\n");
    printVector(B);
    printf("\"C\" matrix:\n");

    for (int i=0; i<DIM*DIM; i++){
      printf("%d", C[i]);
    }
    printf("\n");
    exit(1);

  }

  printf("Input and output matrices are identical, as expected\n");
  printf("\"B\" matrix:\n");
  printVector(B);
  printf("\"C\" matrix:\n");
  printVector(C);
  exit(0);
}


