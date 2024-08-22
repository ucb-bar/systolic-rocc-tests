func.func @tiled_matmul(%arg0: memref<64x64xi8>, %arg1: memref<64x64xi8>, %arg2: memref<64x64xi32>) {
   %c0_i32 = arith.constant 0 : i32
   linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<64x64xi8>, memref<64x64xi8>, i32, i32) outs(%arg2 : memref<64x64xi32>)
   return
}


transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.quantized_matmul">):
  // The actual tiling transformation takes tile sizes as attributes.
   %loop1, %loop2, %loop3, %tiled = transform.structured.tile %arg1 [64, 64, 64]
    : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  transform.yield
}


