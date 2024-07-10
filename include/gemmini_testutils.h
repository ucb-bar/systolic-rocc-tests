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

typedef uint16_t float16_t;


typedef union {
  uint32_t i;
  float    f;
} float_uint32_union_t;


// from https://github.com/AcademySoftwareFoundation/Imath/blob/main/src/Imath/half.h

static inline float NN_halfToFloat(float16_t h) {
  // #if defined(__F16C__)
  //   // NB: The intel implementation does seem to treat NaN slightly
  //   // different than the original toFloat table does (i.e. where the
  //   // 1 bits are, meaning the signalling or not bits). This seems
  //   // benign, given that the original library didn't really deal with
  //   // signalling vs non-signalling NaNs
  //   #ifdef _MSC_VER
  //     /* msvc does not seem to have cvtsh_ss :( */
  //     return _mm_cvtss_f32(_mm_cvtph_ps (_mm_set1_epi16 (h)));
  //   #else
  //     return _cvtsh_ss(h);
  //   #endif
  // #else
  float_uint32_union_t v;
  // this code would be clearer, although it does appear to be faster
  // (1.06 vs 1.08 ns/call) to avoid the constants and just do 4
  // shifts.
  //
  uint32_t hexpmant = ((uint32_t) (h) << 17) >> 4;
  v.i               = ((uint32_t) (h >> 15)) << 31;

  // the likely really does help if most of your numbers are "normal" half numbers
  if ((hexpmant >= 0x00800000)) {
    v.i |= hexpmant;
    // either we are a normal number, in which case add in the bias difference
    // otherwise make sure all exponent bits are set
    if ((hexpmant < 0x0f800000)) {
      v.i += 0x38000000;
    }  
    else {
      v.i |= 0x7f800000;
    }
  }
  else if (hexpmant != 0) {
    // exponent is 0 because we're denormal, don't have to extract
    // the mantissa, can just use as is
    //
    //
    // other compilers may provide count-leading-zeros primitives,
    // but we need the community to inform us of the variants
    uint32_t lc;
    lc = 0;
    while (0 == ((hexpmant << lc) & 0x80000000)) {
      lc += 1;
    }
    lc -= 8;
    // so nominally we want to remove that extra bit we shifted
    // up, but we are going to add that bit back in, then subtract
    // from it with the 0x38800000 - (lc << 23)....
    //
    // by combining, this allows us to skip the & operation (and
    // remove a constant)
    //
    // hexpmant &= ~0x00800000;
    v.i |= 0x38800000;
    // lc is now x, where the desired exponent is then
    // -14 - lc
    // + 127 -> new exponent
    v.i |= (hexpmant << lc);
    v.i -= (lc << 23);
  }
  return v.f;
  // #endif
}

///
/// Convert half to float
///
/// Note: This only supports the "round to even" rounding mode, which
/// was the only mode supported by the original OpenEXR library
///

static inline float16_t NN_floatToHalf(float f) {
  // #if defined(__F16C__)
  //   #ifdef _MSC_VER
  //     // msvc does not seem to have cvtsh_ss :(
  //     return _mm_extract_epi16 (
  //         _mm_cvtps_ph (
  //             _mm_set_ss (f), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
  //         0);
  //   #else
  //     // preserve the fixed rounding mode to nearest
  //     return _cvtss_sh (f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  //   #endif
  // #else
  float_uint32_union_t  v;
  float16_t ret;
  uint32_t e, m, ui, r, shift;

  v.f = f;

  ui  = (v.i & ~0x80000000);
  ret = ((v.i >> 16) & 0x8000);

  // exponent large enough to result in a normal number, round and return
  if (ui >= 0x38800000) {
    // inf or nan
    if (ui >= 0x7f800000) {
      ret |= 0x7c00;
      if (ui == 0x7f800000) {
        return ret;
      }
      m = (ui & 0x7fffff) >> 13;
      // make sure we have at least one bit after shift to preserve nan-ness
      return ret | (uint16_t) m | (uint16_t) (m == 0);
    }

    // too large, round to infinity
    if (ui > 0x477fefff) {
      return ret | 0x7c00;
    }

    ui -= 0x38000000;
    ui = ((ui + 0x00000fff + ((ui >> 13) & 1)) >> 13);
    return ret | (uint16_t) ui;
  }

  // zero or flush to 0
  if (ui < 0x33000001) {
    return ret;
  }

  // produce a denormalized half
  e     = (ui >> 23);
  shift = 0x7e - e;
  m     = 0x800000 | (ui & 0x7fffff);
  r     = m << (32 - shift);
  ret  |= (m >> shift);
  if (r > 0x80000000 || (r == 0x80000000 && (ret & 0x1) != 0)) {
    ret += 1;
  }
  return ret;
// #endif
}


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
#ifndef ELEM_T_IS_FLOAT
      printf("%d ", m[i][j]);
#else
      printf("%x ", elem_t_to_elem_t_bits(m[i][j]));
#endif
    printf("\n");
  }
}

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

static uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbff8);
    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbffc);
    // return *mtime;
}

#undef abs

#endif  // SRC_MAIN_C_GEMMINI_TESTUTILS_H
