// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#define BAREMETAL 1
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define CHECK_RESULT 1

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#define BERT_SCALE 0.8

#ifndef BAREMETAL

#define MAT_DIM_I 128
#define MAT_DIM_K 512
#define MAT_DIM_J 512

#else
#define MAT_DIM_I 5
#define MAT_DIM_K 5
#define MAT_DIM_J 5
#endif

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

int main() {
// #if defined(FAST) || !defined(HAS_NORMALIZATIONS)
//     exit(0);
// #endif

#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    printf("I: %d, J: %d, K: %d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);

    gemmini_flush(0);

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

#if CHECK_RESULT == 1
    // printf("Init A\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        full_A[i][j] = i == j ? NN_floatToHalf(1) : 0; //NN_floatToHalf((rand() % 3) - 1);
      }
    }

    // printf("Init B\n");
    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_B[i][j] = i == j ? NN_floatToHalf(1) : 0; //NN_floatToHalf((rand() % 3) - 1); //(rand() % 3) - 1;
      }
    }

    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_D[i][j] = 0; //NO_BIAS ? 0 : (rand() % 3) - 1;
      }
    }

    printf("Starting slow CPU matmul\n");
    unsigned long cpu_start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)gold,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            NN_floatToHalf(1), NN_floatToHalf(1), NN_floatToHalf(1),
            SOFTMAX, NN_floatToHalf(1), NN_floatToHalf(0.05), false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            CPU);

    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);

#endif

    printf("Starting gemmini matmul\n");
    printf("I: %d, J: %d, K: %d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            NN_floatToHalf(1), NN_floatToHalf(1), NN_floatToHalf(1),
            SOFTMAX, NN_floatToHalf(1), NN_floatToHalf(0.05), false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    gemmini_fence();

    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);
printf("0x%x\n", full_C[0][0]);
printf("0x%x\n", full_C[0][1]);
printf("0x%x\n", full_C[1][0]);
printf("0x%x\n", full_C[1][1]);
//printFPMatrix2(MAT_DIM_I,MAT_DIM_J,full_C);
#if CHECK_RESULT == 1
    if (!full_is_equal(full_C, gold)) {
      printf("C:\n");
      // full_printMatrix(full_C);
      printFPMatrix2(MAT_DIM_I,MAT_DIM_J,full_C);
      printf("Gold:\n");
      // full_printMatrix(gold);
      printFPMatrix2(MAT_DIM_I,MAT_DIM_J,gold);
      printf("\n");

      exit(1);
    }
#endif

  exit(0);
}

