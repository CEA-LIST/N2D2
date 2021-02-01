/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
*/

#include "CublasUtils.hpp"

__global__ void cudaHscal_kernel(int n,
                                 const __half* alpha,
                                 __half* x, int incx)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
#if __CUDA_ARCH__ >= 530
        x[i*incx] = __hmul(*alpha, x[i*incx]);
#else
        x[i*incx] = __float2half(__half2float(*alpha) * __half2float(x[i*incx]));
#endif
    }
}

__global__ void cudaHaxpy_kernel(int n,
                                 const __half* alpha,
                                 const __half* x, int incx,
                                 __half* y, int incy)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
#if __CUDA_ARCH__ >= 530
        y[i*incy] = __hadd(__hmul(*alpha, x[i*incx]), y[i*incy]);
#else
        y[i*incy] = __float2half(__half2float(*alpha) * __half2float(x[i*incx])
                            + __half2float(y[i*incy]));
#endif
    }
}


namespace N2D2 {

template <>
cublasStatus_t  cublasScal<half_float::half>(cublasHandle_t handle, int n,
                                  const half_float::half           *alpha,
                                  half_float::half           *x, int incx)
{
    cudaHscal_kernel<<<(n + 255) / 256, 256>>>
        (n,
        reinterpret_cast<const typename Cuda::cuda_type<half_float::half>::type*>(alpha),
        reinterpret_cast<typename Cuda::cuda_type<half_float::half>::type*>(x),
        incx);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
    return CUBLAS_STATUS_SUCCESS;
}

template <>
cublasStatus_t  cublasScal<float>(cublasHandle_t handle, int n,
                                  const float           *alpha,
                                  float           *x, int incx)
{
    return cublasSscal(handle, n,
        alpha,
        x, incx);
}

template <>
cublasStatus_t  cublasScal<double>(cublasHandle_t handle, int n,
                                  const double           *alpha,
                                  double           *x, int incx)
{
    return cublasDscal(handle, n,
        alpha,
        x, incx);
}

template <>
cublasStatus_t cublasAxpy(cublasHandle_t /*handle*/, int n,
                           const half_float::half           *alpha,
                           const half_float::half           *x, int incx,
                           half_float::half                 *y, int incy)
{
    cudaHaxpy_kernel<<<(n + 255) / 256, 256>>>
        (n,
         reinterpret_cast<const typename Cuda::cuda_type<half_float::half>::type*>(alpha),
         reinterpret_cast<const typename Cuda::cuda_type<half_float::half>::type*>(x),
         incx,
         reinterpret_cast<typename Cuda::cuda_type<half_float::half>::type*>(y),
         incy);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
    return CUBLAS_STATUS_SUCCESS;
}

template <>
cublasStatus_t cublasAxpy(cublasHandle_t handle, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           float                 *y, int incy)
{
    return cublasSaxpy(handle, n,
        alpha,
        x, incx,
        y, incy);
}

template <>
cublasStatus_t cublasAxpy(cublasHandle_t handle, int n,
                           const double           *alpha,
                           const double           *x, int incx,
                           double                 *y, int incy)
{
    return cublasDaxpy(handle, n,
        alpha,
        x, incx,
        y, incy);
}

template <>
cublasStatus_t cublasGemm<__half>(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const __half *alpha,
                           const __half *A, int lda,
                           const __half *B, int ldb,
                           const __half *beta,
                           __half *C, int ldc)
{
    return cublasHgemm(handle,
        transa, transb,
        m, n, k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc);
}

template <>
cublasStatus_t cublasGemm<float>(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc)
{
    return cublasSgemm(handle,
        transa, transb,
        m, n, k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc);
}

template <>
cublasStatus_t cublasGemm<double>(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double *alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           const double *beta,
                           double *C, int ldc)
{
    return cublasDgemm(handle,
        transa, transb,
        m, n, k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc);
}

template <>
cublasStatus_t cublasGemv<__half>(cublasHandle_t handle, cublasOperation_t trans,
                                 int m, int n,
                                 const __half          *alpha,
                                 const __half          *A, int lda,
                                 const __half          *x, int incx,
                                 const __half          *beta,
                                 __half          *y, int incy)
{
    // Using cublasHgemm() because there is no cublasHgemv() yet
    return cublasHgemm(handle,
        trans, CUBLAS_OP_N,
        m, 1, n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy);
}

template <>
cublasStatus_t cublasGemv<float>(cublasHandle_t handle, cublasOperation_t trans,
                                 int m, int n,
                                 const float          *alpha,
                                 const float          *A, int lda,
                                 const float          *x, int incx,
                                 const float          *beta,
                                 float          *y, int incy)
{
    return cublasSgemv(handle, trans,
        m, n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy);
}

template <>
cublasStatus_t cublasGemv<double>(cublasHandle_t handle, cublasOperation_t trans,
                                 int m, int n,
                                 const double          *alpha,
                                 const double          *A, int lda,
                                 const double          *x, int incx,
                                 const double          *beta,
                                 double          *y, int incy)
{
    return cublasDgemv(handle, trans,
        m, n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy);
}

template cublasStatus_t  cublasScal(cublasHandle_t handle, int n,
                                  const half_float::half           *alpha,
                                  half_float::half           *x, int incx);
template cublasStatus_t  cublasScal(cublasHandle_t handle, int n,
                                  const float           *alpha,
                                  float           *x, int incx);
template cublasStatus_t  cublasScal(cublasHandle_t handle, int n,
                                  const double           *alpha,
                                  double           *x, int incx);

template cublasStatus_t cublasAxpy(cublasHandle_t handle, int n,
                           const half_float::half           *alpha,
                           const half_float::half           *x, int incx,
                           half_float::half                 *y, int incy);
template cublasStatus_t cublasAxpy(cublasHandle_t handle, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           float                 *y, int incy);
template cublasStatus_t cublasAxpy(cublasHandle_t handle, int n,
                           const double           *alpha,
                           const double           *x, int incx,
                           double                 *y, int incy);

template cublasStatus_t cublasGemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const __half *alpha,
                           const __half *A, int lda,
                           const __half *B, int ldb,
                           const __half *beta,
                           __half *C, int ldc);
template cublasStatus_t cublasGemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc);
template cublasStatus_t cublasGemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double *alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           const double *beta,
                           double *C, int ldc);

}
