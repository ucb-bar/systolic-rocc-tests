// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_TESTUTILS_H
#define SRC_MAIN_C_GEMMINI_TESTUTILS_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "include/gemmini_params.h"
#include "include/gemmini.h"

#ifdef BAREMETAL
#undef assert
#define assert(expr) \
    if (!(expr)) { \
      printf("Failed assertion: " #expr "\n  " __FILE__ ":%u\n", __LINE__); \
      exit(1); \
    }
#endif

//typedef uint16_t float16_t;




// #define GEMMINI_ASSERTIONS

// Matmul utility functions
static void matmul(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], full_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

static void matmul_short(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], elem_t C[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C[r][c] += A[r][k]*B[k][c];
    }
}

static void matmul_full(elem_t A[DIM][DIM], elem_t B[DIM][DIM], full_t D[DIM][DIM], full_t C_full[DIM][DIM]) {
  // Identical to the other matmul function, but with a 64-bit bias
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

static void matmul_A_transposed(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], full_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[k][r]*B[k][c];
    }
}

static void matmul_short_A_transposed(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], elem_t C[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C[r][c] += A[k][r]*B[k][c];
    }
}

static void matmul_full_A_transposed(elem_t A[DIM][DIM], elem_t B[DIM][DIM], full_t D[DIM][DIM], full_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[k][r]*B[k][c];
    }
}

static void matmul_B_transposed(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], full_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[r][k]*B[c][k];
    }
}

static void matmul_short_B_transposed(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], elem_t C[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C[r][c] += A[r][k]*B[c][k];
    }
}

static void matmul_full_B_transposed(elem_t A[DIM][DIM], elem_t B[DIM][DIM], full_t D[DIM][DIM], full_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[r][k]*B[c][k];
    }
}

static void matmul_AB_transposed(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], full_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[k][r]*B[c][k];
    }
}

static void matmul_short_AB_transposed(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], elem_t C[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C[r][c] += A[k][r]*B[c][k];
    }
}

static void matmul_full_AB_transposed(elem_t A[DIM][DIM], elem_t B[DIM][DIM], full_t D[DIM][DIM], full_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[k][r]*B[c][k];
    }
}

static void matadd(full_t sum[DIM][DIM], full_t m1[DIM][DIM], full_t m2[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      sum[r][c] = m1[r][c] + m2[r][c];
}

// THIS IS A ROUNDING SHIFT! It also performs a saturating cast
static void matshift(full_t full[DIM][DIM], elem_t out[DIM][DIM], int shift) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      // Bitshift and round element
      full_t shifted = ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
      out[r][c] = elem;
#else
      out[r][c] = shifted; // TODO should we also saturate when using floats?
#endif
    }
}

static void matscale(full_t full[DIM][DIM], elem_t out[DIM][DIM], acc_scale_t scale) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      // Bitshift and round element
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

static void matrelu(elem_t in[DIM][DIM], elem_t out[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      out[r][c] = in[r][c] > 0 ? in[r][c] : 0;
}

static void transpose(elem_t in[DIM][DIM], elem_t out[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      out[c][r] = in[r][c];
}

int rand() {
  static uint32_t x = 777;
  x = x * 1664525 + 1013904223;
  return x >> 24;
}


#ifdef ELEM_T_IS_FLOAT
double rand_double() {
    double a = (double)(rand() % 128) / (double)(1 + (rand() % 64));
    double b = (double)(rand() % 128) / (double)(1 + (rand() % 64));
    return a - b;
}
#endif

void NN_printFloat(float v, int16_t num_digits) {
  if (isinf(v)) {
    if (signbit(v)) {
      printf("-inf");
    } else {
      printf("inf");
    }
    return;
  }

  if (v < 0) {
    printf("-");  // Print the minus sign for negative numbers
    v = -v;        // Make the number positive for processing
  }

  // Calculate the integer part of the number
  long int_part = (long)v;
  float fractional_part = v - int_part;

  // Print the integer part
  printf("%ld", int_part);

  if (num_digits > 0) {
    printf("."); // Print the decimal point
  }

  // Handle the fractional part
  while (num_digits > 0) {
    num_digits -= 1;
    fractional_part *= 10;
    int digit = (int)(fractional_part);
    printf("%d", digit);
    fractional_part -= digit;
  }
}

static void printMatrix(elem_t m[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j)
    {
      // printf("%f ", m[i][j]);
      NN_printFloat(m[i][j], 5);
      printf(" ");
    }
// #ifndef ELEM_T_IS_FLOAT
//       printf("%f ", m[i][j]);
// #else
//       printf("%x ", elem_t_to_elem_t_bits(m[i][j]));
// #endif
    printf("\n");
  }
}
// static void printMatrix(elem_t m[DIM][DIM]) {
//   for (size_t i = 0; i < DIM; ++i) {
//     for (size_t j = 0; j < DIM; ++j)
// #ifndef ELEM_T_IS_FLOAT
//       printf("%d ", m[i][j]);
// #else
//       printf("%x ", elem_t_to_elem_t_bits(m[i][j]));
// #endif
//     printf("\n");
//   }
// }

static void printFPMatrix(elem_t m[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      NN_printFloat(NN_halfToFloat((float16_t) (m[i][j])), 5);
      printf(" ");
    }
    printf("\n");
  }
}

static void printFPMatrix2(int rows, int cols, elem_t m[rows][cols]) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      NN_printFloat(NN_halfToFloat((float16_t) (m[i][j])), 5);
      printf(" ");
    }
    printf("\n");
  }
}

static void printMatrixAcc(acc_t m[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j)
#ifndef ELEM_T_IS_FLOAT
      printf("%d ", m[i][j]);
#else
      printf("%x ", acc_t_to_acc_t_bits(m[i][j]));
#endif
    printf("\n");
  }
}

static int is_equal(elem_t x[DIM][DIM], elem_t y[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i)
    for (size_t j = 0; j < DIM; ++j) {
#ifndef ELEM_T_IS_FLOAT
      if (x[i][j] != y[i][j])
#else
      bool isnanx = elem_t_isnan(x[i][j]);
      bool isnany = elem_t_isnan(y[i][j]);

      if (x[i][j] != y[i][j] && !(isnanx && isnany))
#endif
          return 0;
    }
  return 1;
}

static int is_equal_transposed(elem_t x[DIM][DIM], elem_t y[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i)
    for (size_t j = 0; j < DIM; ++j) {
#ifndef ELEM_T_IS_FLOAT
      if (x[i][j] != y[j][i])
#else
      bool isnanx = elem_t_isnan(x[i][j]);
      bool isnany = elem_t_isnan(y[j][i]);

      if (x[i][j] != y[j][i] && !(isnanx && isnany))
#endif
          return 0;
    }
  return 1;
}


static void tiled_global_average_auto(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    enum tiled_matmul_type_t type) {
  if (type == CPU) {
    return global_average_cpu(input, output, batches, channels, dim);
  }

  int channel_tile_size = channels;

  int acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  while (acc_rows > ACC_ROWS) {
    channel_tile_size--;
    acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  }

  tiled_global_average(input, output, batches, channels, dim,
      channel_tile_size);
}

void global_average_cpu(const elem_t * input, elem_t * output,
    int batches, int channels, int dim) {
  const int count = dim * dim;

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel++) {
      acc_t sum = 0;
      //float sum = 0.0;
      for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
          size_t pixel = batch * dim * dim + row * dim + col;

          sum = NN_floatToHalf(NN_halfToFloat(sum) + NN_halfToFloat(input[pixel * channels + channel]));
        }
      }

#ifdef ELEM_T_IS_FLOAT
      output[batch * channels + channel] = NN_floatToHalf(NN_halfToFloat(sum*1.0) / (count*1.0));
#else
      output[batch * channels + channel] = (sum + count/2) / count;
#endif
    }
  }
}

void tiled_global_average(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    int channel_tile_size) {

  gemmini_extended4_config_ld(DIM*sizeof(elem_t), 1, true, 1, 0);
  gemmini_config_ex(0, NO_ACTIVATION, 0);
  gemmini_extended_config_st(0, NO_ACTIVATION, NN_floatToHalf(1.0 / (dim*dim)));

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel += channel_tile_size) {
      const int tile_size = channel + channel_tile_size <= channels ?
        channel_tile_size : channels - channel;

      sp_tiled_global_average(input + batch * dim * dim * channels + channel,
          output + batch * channels + channel,
          batches, channels, dim, tile_size);
    }
  }
}



// This is a GNU extension known as statment expressions
#define MAT_IS_EQUAL(dim_i, dim_j, x, y) \
    ({int result = 1; \
      for (size_t i = 0; i < dim_i; i++) \
        for (size_t j = 0; j < dim_j; ++j) { \
          if (x[i][j] != y[i][j]) { \
            result = 0; \
            break; \
          } \
        } \
      result;})



#undef abs

#endif  // SRC_MAIN_C_GEMMINI_TESTUTILS_H
