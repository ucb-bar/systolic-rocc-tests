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
#define D_scale_factor 1.0
#define scale 1.0
#define stride


int main() {
const A_row_stride = DIM; //ele in a row of DRAM
const B_row_stride = 1; //ele in a row of DRAM
const C_row_stride = DIM;//ele in a row of DRAM
const D_row_stride = 0;
const A_stride = 1; //ele in a row of SPAD
const C_stride = 1;//ele in a row of SPAD
const sizeof_C = sizeof(elem_t);
const tile_I = 5;//TODO; //num of DIM*DIM in A
const tile_J = 1; 
const tile_K = 1;// TODO; // num of DIM in B
static uint64_t startTimestamp;

    for (int tile_I = 1; tile_I <= 6; tile_I++) {  // Adjust the scaling range as needed
        for (int tile_K = 1; tile_K <= 6; tile_K++) {  // Adjust the scaling range as needed

            elem_t A[tile_I*DIM*DIM][tile_K*DIM];
            elem_t B[tile_K*DIM];
            elem_t C[tile_I*DIM*DIM];
            acc_t output[tile_I*DIM*DIM];
            elem_t output_gold[tile_I*DIM*DIM];

            // Initialize matrices A and vector B
            for (int i = 0; i < tile_I*DIM*DIM; i++) {
                for (int j = 0; j < tile_K*DIM; j++) {
                    A[i][j] = (rand() % 64) - 32;
                }
            }
            for (int i = 0; i < tile_K*DIM; i++) {
                B[i] = (rand() % 64) - 32;
            }

            // Start timing the matrix-vector multiplication
            printf("Starting gemmini matvec with tile_I=%d, tile_K=%d\n", tile_I, tile_K);
            unsigned long start = read_cycles();
            // sp_tiled_matmul_ws(A, B, NULL, NULL, A_scale_factor, B_scale_factor, D_scale_factor,  tile_I*DIM,1, tile_K, 0, 0, 0, A_row_stride, B_row_stride, D_row_stride, C_row_stride, false,false, false, false,  false, false, 0, 1,1);
            sp_tiled_matvec_ws(A, B, C, A_scale_factor, B_scale_factor, DIM,1, 1, 0, 0, 0, A_row_stride, B_row_stride, C_row_stride, false, false);

            unsigned long end = read_cycles();

            // Record the number of cycles taken
            printf("Cycles taken for tile_I=%d, tile_K=%d: %lu\n", tile_I, tile_K, end - start);
            gemmini_fence();

            // // Compute the expected output for verification
            // for (int i = 0; i < tile_I*DIM*DIM; i++) {
            //     output[i] = 0;
            //     for (int j = 0; j < tile_K*DIM; j++) {
            //         output[i] += A[i][j] * B[j];
            //     }
            //     output[i] = output[i] * scale;
            //     output[i] = output[i] > elem_t_max ? elem_t_max : output[i];
            //     output[i] = output[i] < elem_t_min ? elem_t_min : output[i];
            //     output_gold[i] = output[i];
            }
 
            // // Check and print results
            // printf("Check whether \"In\" and \"Out\" matrices are identical\n");
            // if (!is_equal_vector(output_gold, C, tile_I*DIM*DIM)) {
            //     printf("C and output matrices are different!\n");
            //     for (int i = 0; i < tile_I*DIM*DIM; i++) {
            //         printf("%d ", C[i]);
            //     }
            //     printf("\n");
            //     exit(1);
            // }

            // printf("Input and output matrices are identical, as expected\n");
            // for (int i = 0; i < tile_I*DIM*DIM; i++) {
            //     printf("%d ", output_gold[i]);
            // }
            // printf("\n");
        }
    }


// int main() {
//     const int A_row_stride = DIM; // Elements in a row of DRAM for A
//     const int B_row_stride = 1;   // Elements in a row of DRAM for B
//     const int C_row_stride = DIM; // Elements in a row of DRAM for C
//     const int A_stride = 1;       // Elements in a row of SPAD for A
//     const int C_stride = 1;       // Elements in a row of SPAD for C
//     const int sizeof_C = sizeof(elem_t);

//     for (int tile_I = 1; tile_I <= 64; tile_I++) {   // Adjust upper limit as needed for scaling
//         for (int tile_K = 1; tile_K <= 64; tile_K++) {   // Adjust upper limit as needed for scaling

//             elem_t A[tile_I*DIM*DIM][tile_K*DIM];
//             elem_t B[tile_K*DIM];
//             elem_t C[tile_I*DIM*DIM];
//             acc_t output[tile_I*DIM*DIM];
//             elem_t output_gold[tile_I*DIM*DIM];

//             // Initialize matrix A
//             for (int i = 0; i < tile_I*DIM*DIM; i++) {
//                 for (int j = 0; j < tile_K*DIM; j++) {
//                     A[i][j] = (rand() % 64) - 32;
//                 }
//             }

//             // Initialize vector B
//             for (int i = 0; i < tile_K*DIM; i++) {
//                 B[i] = (rand() % 64) - 32;
//             }

//             // Perform the matrix-vector multiplication
//             for (int i = 0; i < tile_I*DIM*DIM; i++) {
//                 output[i] = 0;
//                 for (int j = 0; j < tile_K*DIM; j++) {
//                     output[i] += A[i][j] * B[j];
//                 }
//                 output[i] = output[i] * scale;
//                 output[i] = output[i] > elem_t_max ? elem_t_max : output[i];
//                 output[i] = output[i] < elem_t_min ? elem_t_min : output[i];
//                 output_gold[i] = output[i];
//             }

//             printf("Matrix-vector multiplication with tile_I=%d, tile_K=%d completed.\n", tile_I, tile_K);
//             // Here you can add the Gemmini execution code and checks, if necessary.
//         }
//     }

//     // Cleanup or further processing can be added here
// }
