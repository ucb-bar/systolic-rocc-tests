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

#define N 2

#if (N*DIM) > (BANK_NUM*BANK_ROWS)
#error not enough scratchpad space
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  int cfgid = 0;
  int i = 0;
  bool acquired = rr_acquire_single(cfgid, i);

  if(acquired){
    printf("gemmini %d acquired to cfgid %d\n", i, cfgid);
  }

  int cfgid1 = 1;
  int i1 = 1;
  acquired = rr_acquire_single(cfgid1, i1);

  if(acquired){
    printf("gemmini %d acquired to cfgid %d\n", i1, cfgid1);
  }

  rr_set_opc(XCUSTOM_ACC, cfgid);

  // printf("Flush\n");
  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  static elem_t In[N][DIM][DIM] row_align(1);
  static elem_t Out[N][DIM][DIM] row_align(1);

  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j)
        In[n][i][j] = i*DIM + j + n;

  for (size_t n = 0; n < N-1; ++n) {
    printf("Mvin %d\n", n);
    gemmini_mvin(In[n], n*DIM);
    printf("Mvout %d\n", n);
    gemmini_mvout(Out[n], n*DIM);
    printf("Mvin %d\n", n+1);
    gemmini_mvin(In[(n+1)], (n+1)*DIM);
    //printf("Mvout spad->spad%d\n", n);
    //gemmini_mvout(0x1100000, (n+1)*DIM);
    printf("Mvout %d\n", n+1);
    gemmini_mvout(Out[n], (n+1)*DIM);
    //printf("Mvin spad->spad%d\n", n);
    //gemmini_mvin(0x1100000, (n+2)*DIM);
  }

  printf("Fence");
  gemmini_fence();
  rr_fence(cfgid);

  for (size_t n = 0; n < N-1; ++n)
    if (!is_equal(In[n+1], Out[n])) {
      printf("Matrix %u:\n", n);
      printMatrix(In[n+1]);
      printf("Matrix %u output:\n", n);
      printMatrix(Out[n]);
      printf("\n");

      exit(1);
    }

  rr_set_opc(XCUSTOM_ACC, cfgid1);

  // printf("Flush\n");
  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  for (size_t n = 0; n < N-1; ++n) {
    printf("Mvin %d\n", n);
    gemmini_mvin(In[n], n*DIM);
    printf("Mvout %d\n", n);
    gemmini_mvout(Out[n], n*DIM);
    printf("Mvin %d\n", n+1);
    gemmini_mvin(In[(n+1)], (n+1)*DIM);
    printf("Mvout spad->spad%d\n", n+1);
    gemmini_mvout(0x1000000, (n+1)*DIM);
    rr_fence(cfgid1);
    printf("Mvout %d\n", n+1);
    gemmini_mvout(Out[n], (n+1)*DIM);
    gemmini_mvin(0x1000000, (n+2)*DIM);
    printf("Mvout %d\n", n+2);
    gemmini_mvout(Out[n], (n+2)*DIM);
  }

  printf("Fence");
  gemmini_fence();
  rr_fence(cfgid1);

  rr_release(cfgid);
  rr_release(cfgid1);

  printf("Done");

  for (size_t n = 0; n < N-1; ++n)
    if (!is_equal(In[n+1], Out[n])) {
      printf("Matrix %u:\n", n);
      printMatrix(In[n+1]);
      printf("Matrix %u output:\n", n);
      printMatrix(Out[n]);
      printf("\n");

      exit(1);
    }

  exit(0);
}