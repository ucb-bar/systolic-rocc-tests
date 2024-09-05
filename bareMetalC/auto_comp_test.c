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

#define ACTIVATION NO_ACTIVATION

// #define FULL_BIAS_WIDTH 0
// #if FULL_BIAS_WIDTH
// typedef acc_t ACC_T;
// #else
// typedef elem_t ACC_T;
// #endif

#define MAKESTR(NAME) #NAME
#define XMAKESTR(NAME) MAKESTR(NAME)

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#define NO_BIAS 1
#define REPEATING_BIAS 0
#define SUB_BIAS 0

#define A_MATRIX_NAME A
#define B_MATRIX_NAME B
#define C_MATRIX_NAME C
#if NO_BIAS==0
  #define D_MATRIX_NAME D
#endif

#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512

#if A_TRANSPOSE==1
  #define A_ROWS MAT_DIM_K
  #define A_COLS MAT_DIM_I
#else
  #define A_ROWS MAT_DIM_I
  #define A_COLS MAT_DIM_K
#endif

#if B_TRANSPOSE==1
  #define B_ROWS MAT_DIM_J
  #define B_COLS MAT_DIM_K
#else
  #define B_ROWS MAT_DIM_K
  #define B_COLS MAT_DIM_J
#endif

#define CHECK_RESULT 1
#undef FAST
// #define FAST

void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

// void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
//   for (size_t r = 0; r < MAT_DIM_I; r++)
//     for (size_t c = 0; c < MAT_DIM_J; c++) {
//       C_full[r][c] = D[r][c];
//       for (size_t k = 0; k < MAT_DIM_K; k++)
//         C_full[r][c] += A[r][k]*B[k][c];
//     }
// }

void A_printMatrix(elem_t m[A_ROWS][A_COLS]) {
  for (size_t i = 0; i < A_ROWS; ++i) {
    for (size_t j = 0; j < A_COLS; ++j)
      printf("%d ", (int) m[i][j]);
    printf("\n");
  }
}

void B_printMatrix(elem_t m[B_ROWS][B_COLS]) {
  for (size_t i = 0; i < B_ROWS; ++i) {
    for (size_t j = 0; j < B_COLS; ++j)
      printf("%d ", (int) m[i][j]);
    printf("\n");
  }
}

void D_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", (int) m[i][j]);
    printf("\n");
  }
}

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", (int) m[i][j]);
    printf("\n");
  }
}

// void my_printMatrix(elem_t **m, int I, int J) {
//   for (size_t i = 0; i < I; ++i) {
//     for (size_t j = 0; j < J; ++j)
//       printf("%d ", m[i][j]);
//     printf("\n");
//   }
// }

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Scale element
      full_t scaled = ACC_SCALE(full[r][c], scale);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    static elem_t A_MATRIX_NAME[A_ROWS][A_COLS] row_align(1);
    static elem_t B_MATRIX_NAME[B_ROWS][B_COLS] row_align(1);
#if NO_BIAS==0
    static acc_t D_MATRIX_NAME[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
#endif
    static acc_t gold[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
    static acc_t C_MATRIX_NAME[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

#ifdef FAST
#define RAND 1
#else
#define RAND rand()
#endif
    // printf("Init A\n");
    for (size_t i = 0; i < A_ROWS; i++) {
      for (size_t j = 0; j < A_COLS; j++) {
        A_MATRIX_NAME[i][j] = RAND % 2;
      }
    }

    // printf("Init B\n");
    for (size_t i = 0; i < B_ROWS; i++) {
      for (size_t j = 0; j < B_COLS; j++) {
        B_MATRIX_NAME[i][j] = RAND % 2;
      }
    }

#if NO_BIAS==0
    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; i++) {
      for (size_t j = 0; j < MAT_DIM_J; j++) {
        D_MATRIX_NAME[i][j] = RAND % 2;
      }
    }
#endif

    // Baseline implementation
#if NO_BIAS==0
    tiled_matmul_outer_eigen_bias(A_MATRIX_NAME, B_MATRIX_NAME, D_MATRIX_NAME, gold, MAT_DIM_I, MAT_DIM_K, MAT_DIM_J, A_TRANSPOSE, B_TRANSPOSE, SUB_BIAS);
#else
    tiled_matmul_outer_eigen(A_MATRIX_NAME, B_MATRIX_NAME, gold, MAT_DIM_I, MAT_DIM_K, MAT_DIM_J, A_TRANSPOSE, B_TRANSPOSE);
#endif

    // Generated implementation
    // SUBSTITUTE HERE

    // Configuration for the systolic array operations
    config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
    config_st(512 * sizeof(float)); // C has 512 columns in DRAM
    
    // Configuration for loading matrices into scratchpad memory
    config_ld(512 * sizeof(float), 1, 0); // A matrix has 512 columns in DRAM
    config_ld(512 * sizeof(float), 1, 1); // B matrix has 512 columns in DRAM

    // Addresses in scratchpad and accumulator
    static uint32_t A_sp_addr = 0;       // Address in scratchpad for A
    static uint32_t B_sp_addr = 0x2000;  // Address in scratchpad for B
    static uint32_t C_acc_addr = 1 << 31; // Address in accumulator for C

    // Tile sizes to fit into scratchpad and accumulator
    const int TILE_SIZE_I = 64; // Tile size for I dimension (A and C)
    const int TILE_SIZE_K = 64; // Tile size for K dimension (A and B)
    const int TILE_SIZE_J = 64; // Tile size for J dimension (B and C)

    // Perform the matrix multiplication in tiles
    for (int i = 0; i < 512; i += TILE_SIZE_I) {
        for (int j = 0; j < 512; j += TILE_SIZE_J) {
            for (int k = 0; k < 512; k += TILE_SIZE_K) {

                // Move A and B matrices into the scratchpad
                for (int ii = 0; ii < TILE_SIZE_I; ii += 4) {
                    for (int kk = 0; kk < TILE_SIZE_K; kk += 4) {
                        uint32_t A_tile_addr = A_sp_addr + (ii * TILE_SIZE_K + kk * 4) / 4;
                        mvin(&A[i + ii][k + kk], A_tile_addr, 4, 4);
                    }
                }
                for (int jj = 0; jj < TILE_SIZE_J; jj += 4) {
                    for (int kk = 0; kk < TILE_SIZE_K; kk += 4) {
                        uint32_t B_tile_addr = B_sp_addr + (kk * TILE_SIZE_J + jj * 4) / 4;
                        mvin2(&B[k + kk][j + jj], B_tile_addr, 4, 4);
                    }
                }

                // Perform the matrix multiplication on the tiles
                for (int ii = 0; ii < TILE_SIZE_I; ii += 4) {
                    for (int jj = 0; jj < TILE_SIZE_J; jj += 4) {
                        for (int kk = 0; kk < TILE_SIZE_K; kk += 4) {
                            uint32_t A_block_addr = A_sp_addr + (ii * TILE_SIZE_K + kk * 4) / 4;
                            uint32_t B_block_addr = B_sp_addr + (kk * TILE_SIZE_J + jj * 4) / 4;
                            uint32_t C_block_addr = C_acc_addr + (ii * TILE_SIZE_J + jj * 4) / 4;

                            if (k == 0 && kk == 0) {
                                preload(B_block_addr, C_block_addr, 4, 4, 4, 4);
                                compute_preloaded(A_block_addr, 0xffffffff, 4, 4, 4, 4);
                            } else {
                                preload(B_block_addr, C_block_addr | (1 << 30), 4, 4, 4, 4);
                                compute_preloaded(A_block_addr, 0xffffffff, 4, 4, 4, 4);
                            }
                        }
                    }
                }
            }

            // Move the result from the accumulator back to DRAM
            for (int ii = 0; ii < TILE_SIZE_I; ii += 4) {
                for (int jj = 0; jj < TILE_SIZE_J; jj += 4) {
                    uint32_t C_out_addr = C_acc_addr + (ii * TILE_SIZE_J + jj * 4) / 4;
                    mvout(&C[i + ii][j + jj], C_out_addr, 4, 4);
                }
            }
        }
    }

    printf("%d\n", C[0][0]);
    // Ensure all operations are completed before proceeding
    fence();

    // SUBSTITUTE END

    printf("%s:\n", XMAKESTR(A_MATRIX_NAME));
    A_printMatrix(A_MATRIX_NAME);
    printf("%s:\n", XMAKESTR(B_MATRIX_NAME));
    B_printMatrix(B_MATRIX_NAME);
#if NO_BIAS==0
    printf("%s:\n", XMAKESTR(D_MATRIX_NAME));
    D_printMatrix(D_MATRIX_NAME);
#endif

    // Check result
    if (!full_is_equal(C_MATRIX_NAME, gold)) {
      printf("Incorrect result\n");
      printf("%s:\n", XMAKESTR(C_MATRIX_NAME));
      full_printMatrix(C_MATRIX_NAME);
      printf("%s was supposed to be:\n", XMAKESTR(C_MATRIX_NAME));
      full_printMatrix(gold);
      printf("\n");

      exit(1);
    } else {
      printf("Correct result\n");
    }

}
