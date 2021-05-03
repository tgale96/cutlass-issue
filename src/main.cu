#include <cassert>
#include <iostream>

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "gpu_timer.h"

using gemm_mixed_128x256_32x3_tn_align8_base =
  typename ::cutlass::gemm::kernel::DefaultGemmUniversal<
    ::cutlass::half_t, ::cutlass::layout::RowMajor, ::cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    ::cutlass::half_t, ::cutlass::layout::ColumnMajor, ::cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    ::cutlass::half_t, ::cutlass::layout::RowMajor,
    float,
    ::cutlass::arch::OpClassTensorOp,
    ::cutlass::arch::Sm80,
    ::cutlass::gemm::GemmShape<128, 256, 32>,
    ::cutlass::gemm::GemmShape<64, 64, 32>,
    ::cutlass::gemm::GemmShape<16, 8, 16>,
    ::cutlass::epilogue::thread::LinearCombination<
      ::cutlass::half_t,
      8,
      float,
      float
    >,
    ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    ::cutlass::arch::OpMultiplyAdd
>::GemmKernel;  

// Define named type
struct gemm_mixed_128x256_32x3_tn_align8 : 
  public gemm_mixed_128x256_32x3_tn_align8_base { };
    
int main() {
  int M = 8192, N = 8192, K = 8192;
  ::cutlass::half_t *A, *B, *C;

  size_t bytes = M * K * sizeof(::cutlass::half_t);
  cudaError_t custatus = cudaMalloc(&A, bytes);
  assert(custatus == cudaSuccess);
  cudaError_t custatus = cudaMalloc(&B, bytes);
  assert(custatus == cudaSuccess);  
  cudaError_t custatus = cudaMalloc(&C, bytes);
  assert(custatus == cudaSuccess);  
  
  using Gemm = ::cutlass::gemm::device::GemmUniversalAdapter<
    gemm_mixed_128x256_32x3_tn_align8_base>;

  Gemm::Arguments args(::cutlass::gemm::GemmUniversalMode::kGemm,
		       {M, N, K},
		       /*batch_count=*/1,
		       {1.0f, 0.0f},
		       A, B, C, C,
		       0, 0, 0, 0,
		       /*lda=*/K,
		       /*ldb=*/K,
		       /*ldc=*/N,
		       /*ldd=*/N);  
  args.transposed_problem();

  Gemm gemm_operator;
  ::cutlass::Status status = gemm_operator.initialize(args, nullptr, stream);
  assert(status == ::cutlass::Status::kSuccess);
  
  int sleep_duration = 50;
  int warmup_iterations = 10;
  int iterations = 100;
  GpuTimer timer;

  // Sleep.
  usleep(sleep_duration * 1e3);

  // Warmup.
  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm_operator();
    assert(status == ::cutlass::Status::kSuccess);    
  }

  // Timed iterations.
  timer.start();    
  for (int i = 0; i < iterations; ++i) {
    status = gemm_operator();
    assert(status == ::cutlass::Status::kSuccess);    
  }
  timer.stop_and_wait();
  double ms = timer.duration(iterations);
  int64_t flops = (int64_t)kDimM * kDimK * kDimN * 2;

  std::cout << "ms = " << ms << std::endl;  
  std::cout << "flops = " << flops << std::endl;
  std::cout << "TFLOPs = " << flops / (ms / 1e3) / 1e12 << std::endl;
}  

  
