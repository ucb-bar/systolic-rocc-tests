// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include <time.h>
#include "include/gemmini_testutils.h"

#define DIM 16

#ifdef FAST
#define AINIT RELU
#define SINIT 12
#define N 1
#else
#define AINIT NO_ACTIVATION
#define SINIT 0
#define N 1
#endif

void operands(int c, int *a, int *b, int *d) {
  *d = c % N;
  *b = (c / N) % N;
  *a = c / (N * N);
}

void full_matmul(elem_t A[DIM][DIM], elem_t B[DIM][DIM], acc_t D[DIM][DIM], full_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; ++r)
    for (size_t c = 0; c < DIM; ++c) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; ++k) {
        C_full[r][c] = NN_floatToHalf(NN_halfToFloat(C_full[r][c]) + NN_halfToFloat(A[r][k]) * NN_halfToFloat(B[k][c]));
      }
    }
}

void full_printMatrix(elem_t m[DIM][DIM][DIM]) {
    for (size_t n = 0; n < N; ++n)
      for (size_t i = 0; i < DIM; ++i)
        for (size_t j = 0; j < DIM; ++j){
          printf("%d ", m[n][i][j]);
        printf("\n");
  }
}

int full_is_equal(elem_t x[DIM][DIM][DIM], elem_t y[DIM][DIM][DIM]) {
  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j)
        if (x[n][i][j] != y[n][i][j])
          return 0;
    return 1;
}


int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  static elem_t ZERO[DIM][DIM] = {0};

  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));

  for (int activation = AINIT; activation <= RELU; ++activation) {
#ifdef ACC_SCALE_T_IS_FLOAT
    //for (acc_scale_t scale = 0; scale <= 1.5; scale += 0.5) {
    for (float scale = 0; scale <= 1.5; scale += 0.5) {
#else
    for (acc_scale_t scale = SINIT; scale <= 12; scale += 4) {
#endif
      static elem_t A[N][DIM][DIM] row_align(1);
      static elem_t B[N][DIM][DIM] row_align(1);
      static elem_t D[N][DIM][DIM] row_align(1);

      static elem_t C[N * N * N][DIM][DIM] row_align(1);
      static full_t gold_full[N * N * N][DIM][DIM];
      static elem_t gold[N * N * N][DIM][DIM];

      static int preload[N * N * N] = {1};
      for (int i = 1; i < N * N * N; ++i)
        preload[i] = 1;//rand() % 2;

      static int add_to_zeros[N * N * N];
      for (int i = 0; i < N * N * N; ++i)
        add_to_zeros[i] = 1;//rand() % 2;

      static int accumulate[N * N * N] = {0};
      for (int i = 1; i < N * N * N; ++i)
        accumulate[i] = 0;//rand() % 2;

      static int no_output[N * N * N];
      for (int i = 0; i < N * N * N - 1; ++i)
        no_output[i] = accumulate[i + 1];
      no_output[N * N * N - 1] = 0;

      for (size_t n = 0; n < N; ++n) {
        for (size_t i = 0; i < DIM; ++i) {
          for (size_t j = 0; j < DIM; ++j) {
            A[n][i][j] = NN_floatToHalf((float)(1));//NN_floatToHalf((float)((rand() % 64) - 32));
            B[n][i][j] = NN_floatToHalf((float)((rand() % 64) - 32));
            D[n][i][j] = NN_floatToHalf((float)((rand() % 64) - 32));
          }
        }
      }

      // printf("Matrix A:\n");
      // for (size_t n = 0; n < N; ++n)
      //   printFPMatrix(A[n]);

      // printf("Matrix B:\n");
      // for (size_t n = 0; n < N; ++n)
      //   printFPMatrix(B[n]);

      // printf("Matrix D:\n");
      // for (size_t n = 0; n < N; ++n)
      //   printFPMatrix(D[n]);

      for (size_t g = 0; g < N * N * N; ++g) {
        int a, b, d;
        operands(g, &a, &b, &d);

        for (int last_g = g; last_g >= 0; --last_g) {
          int tmp_a, tmp_d;
          if (preload[last_g]) {
            operands(last_g, &tmp_a, &b, &tmp_d);
            break;
          }
        }

        if (add_to_zeros[g])
          full_matmul(A[a], B[b], ZERO, gold_full[g]);
        else
          full_matmul(A[a], B[b], D[d], gold_full[g]);

        if (accumulate[g])
          matadd(gold_full[g], gold_full[g - 1], gold_full[g]);
      }

      for (size_t g = 0; g < N * N * N; ++g) {
        matscale(gold_full[g], gold[g], NN_floatToHalf(scale));
        if (activation == RELU)
          matrelu(gold[g], gold[g]);
      }

      uint32_t A_addr = 0;
      uint32_t B_addr = N * DIM;
      uint32_t D_addr = 2 * N * DIM;
      uint32_t C_addr_acc = 1 << (ADDR_LEN - 1);

      uint32_t C_addrs[N * N * N];
      for (size_t c = 0; c < N * N * N; ++c)
        C_addrs[c] = C_addr_acc + c * DIM;
      for (size_t c = 0; c < N * N * N; ++c) {
        int last_c;
        for (last_c = c; last_c >= 0; --last_c)
          if (!accumulate[last_c])
            break;
        if (c != last_c)
          C_addrs[c] = C_addrs[last_c] | (1 << (ADDR_LEN - 2));
      }

      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(A[n], A_addr + n * DIM);

      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(B[n], B_addr + n * DIM);

      for (size_t n = 0; n < N; ++n)
        if (n == N - 1) {
          gemmini_mvin(D[n], D_addr + n * DIM);
        } else {
          gemmini_mvin(D[n], D_addr + n * DIM);
        }

      gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
      gemmini_extended_config_st(DIM * sizeof(elem_t), activation, NN_floatToHalf(scale));

      for (size_t c = 0; c < N * N * N; ++c) {
        int a, b, d;
        operands(c, &a, &b, &d);

        uint32_t d_addr = D_addr + d * DIM;
        if (add_to_zeros[c])
          d_addr = GARBAGE_ADDR;

        if (!preload[c]) {
          gemmini_preload_zeros(C_addrs[c]);
          gemmini_compute_accumulated(A_addr + a * DIM, d_addr);
        } else {
          gemmini_preload(B_addr + b * DIM, C_addrs[c]);
          gemmini_compute_preloaded(A_addr + a * DIM, d_addr);
        }
      }

      for (size_t c = 0; c < N * N * N; ++c)
        if (!no_output[c]) {
          gemmini_mvout(C[c], C_addrs[c] & ~(1 << (ADDR_LEN - 2)));
        }

      gemmini_fence();

      printf("Matrix C:\n");
      for (int n = 0; n < N * N * N; ++n)
        printFPMatrix(C[n]);

      printf("Gold matrix:\n");
      for (int n = 0; n < N * N * N; ++n)
        printFPMatrix(gold[n]);

      for (int n = 0; n < N * N * N; ++n)
        if (!no_output[n] && !full_is_equal(C[n], gold[n])) {
          printf("activation: %d, scale: %d\n", activation, scale);
          printf("Actual (%d):\n", n);
          printFPMatrix(C[n]);
          printf("\nGold:\n");
          printFPMatrix(gold[n]);
          exit(1);
        }
    }
  }

  printf("PASS\n");
  exit(0);
}