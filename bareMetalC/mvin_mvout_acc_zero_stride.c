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

#define N 4

#if (N*DIM) > ACC_ROWS
#error not enough accumulator space
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  gemmini_flush(0);

  static acc_t In[N][DIM] row_align_acc(1);
  static elem_t Out[N][DIM][DIM] row_align(1);
  static elem_t Out_gold[N][DIM][DIM];

  for (size_t n = 0; n < N; ++n)
    for (size_t j = 0; j < DIM; ++j) {
#ifndef ELEM_T_IS_FLOAT
      In[n][j] = 0;

      int bytes = rand() % 2 ? sizeof(acc_t) : sizeof(elem_t);
      for (size_t b = 0; b < bytes; ++b) {
        In[n][j] = NN_floatToHalf(1); //|= (rand() % 255) << (b*8);
      }
#else
      acc_t_bits data;

      do {
        data = 0;

        int bytes = rand() % 2 ? sizeof(acc_t) : sizeof(elem_t);
        for (size_t b = 0; b < bytes; ++b) {
          data |= (uint64_t)(rand() % 255) << (b*8);
        }

        In[n][j] = NN_floatToHalf(1); //acc_t_bits_to_acc_t(data);
      } while (acc_t_isnan(In[n][j]));
#endif
    }

  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j) {
        Out_gold[n][i][j] = NN_floatToHalf(ACC_SCALE(NN_halfToFloat(In[n][j]), ACC_SCALE_IDENTITY));
      }

  const uint32_t acc_addr = 1 << (ADDR_LEN-1);

  // printf("Config\n");
  gemmini_config_ld(0);
  gemmini_config_ex(0, NO_ACTIVATION, 0);
  gemmini_extended_config_st(DIM*sizeof(elem_t), NO_ACTIVATION, NN_floatToHalf(1));

  // printf("Mvin and mvout\n");
  for (size_t n = 0; n < N; ++n) {
    // printf("Mvin n: %u\n", n);
    gemmini_mvin(In[n], acc_addr + n*DIM);
    // printf("Mvout n: %u\n", n);
    gemmini_mvout(Out[n], acc_addr + n*DIM);
  }

  // printf("Fence\n");
  gemmini_fence();

  // printf("Check\n");
  for (size_t n = 0; n < N; ++n)
    if (!is_equal(Out[n], Out_gold[n])) {
      printf("Matrix %u:\n", n);
      printFPMatrix2(1, DIM, In[n]);
      printf("Matrix %u output:\n", n);
      printFPMatrix2(1, DIM, Out[n]);
      printf("Matrix %u gold output:\n", n);
      printFPMatrix2(1, DIM, Out_gold[n]);
      printf("\n");

      exit(1);
    }
  printf("Pass\n");
  exit(0);
}

