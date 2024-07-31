; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @tiled_matmul(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
  br label %22

22:                                               ; preds = %49, %21
  %23 = phi i64 [ %50, %49 ], [ 0, %21 ]
  %24 = icmp slt i64 %23, 128
  br i1 %24, label %25, label %51

25:                                               ; preds = %22
  %26 = sub i64 128, %23
  %27 = icmp sgt i64 %26, 80
  %28 = select i1 %27, i64 80, i64 %26
  br label %29

29:                                               ; preds = %47, %25
  %30 = phi i64 [ %48, %47 ], [ 0, %25 ]
  %31 = icmp slt i64 %30, 128
  br i1 %31, label %32, label %49

32:                                               ; preds = %29
  %33 = sub i64 128, %30
  %34 = icmp sgt i64 %33, 96
  %35 = select i1 %34, i64 96, i64 %33
  br label %36

36:                                               ; preds = %39, %32
  %37 = phi i64 [ %46, %39 ], [ 0, %32 ]
  %38 = icmp slt i64 %37, 128
  br i1 %38, label %39, label %47

39:                                               ; preds = %36
  %40 = mul i64 %23, 128
  %41 = add i64 %40, %37
  %42 = mul i64 %37, 128
  %43 = add i64 %42, %30
  %44 = mul i64 %23, 128
  %45 = add i64 %44, %30
  call void @sp_tiled_matmul_ws(ptr %0, ptr %1, i64 %41, i64 %28, i64 128, i64 128, i64 1, ptr %7, ptr %8, i64 %43, i64 128, i64 %35, i64 128, i64 1, ptr %14, ptr %15, i64 %45, i64 %28, i64 %35, i64 128, i64 1)
  %46 = add i64 %37, 128
  br label %36

47:                                               ; preds = %36
  %48 = add i64 %30, 96
  br label %29

49:                                               ; preds = %29
  %50 = add i64 %23, 80
  br label %22

51:                                               ; preds = %22
  ret void
}

define void @_mlir_ciface_tiled_matmul(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %0, align 8
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 3, 1
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 4, 1
  %12 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 0
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 1
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 2
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 0
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 1
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 0
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 1
  %20 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %21 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, 0
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, 2
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, 3, 0
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, 3, 1
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, 4, 0
  %27 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, 4, 1
  call void @tiled_matmul(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, ptr %21, ptr %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27)
  ret void
}

define private void @sp_tiled_matmul_ws(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %1, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %2, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %3, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %5, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %4, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %6, 4, 1
  %29 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, ptr %29, align 8
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %7, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, ptr %8, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %9, 2
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %10, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %12, 4, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %11, 3, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %13, 4, 1
  %37 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %37, align 8
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %14, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, ptr %15, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %16, 2
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %17, 3, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %19, 4, 0
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %18, 3, 1
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %20, 4, 1
  %45 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, ptr %45, align 8
  call void @_mlir_ciface_sp_tiled_matmul_ws(ptr %29, ptr %37, ptr %45)
  ret void
}

declare void @_mlir_ciface_sp_tiled_matmul_ws(ptr, ptr, ptr)

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

