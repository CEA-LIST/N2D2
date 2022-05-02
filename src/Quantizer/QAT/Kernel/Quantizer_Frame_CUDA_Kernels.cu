/**
 * (C) Copyright 2020 CEA LIST. All Rights Reserved.
 *  Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
 *                  David BRIAND (david.briand@cea.fr)
 *                  Inna KUCHER (inna.kucher@cea.fr)
 *                  Olivier BICHLER (olivier.bichler@cea.fr)
 *                  Vincent TEMPLIER (vincent.templier@cea.fr)
 * 
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 * 
 */

#include "Quantizer/QAT/Kernel/Quantizer_Frame_CUDA_Kernels.hpp"
#include "CudaUtils.hpp"
#include <stdlib.h>
#include <math.h>

/* Macros */
#define imin(a,b) (a<b?a:b)


__global__ void cudaH_sum_kernel(__half* x,  
                                 __half* sum, 
                                 unsigned int size)
{
    //256 - threadsPerBlock
    __shared__ __half cache[256];

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    int cacheIndex = threadIdx.x;

    __half temp = __float2half(0.0f);
    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
    temp = __hadd(temp, x[i]);
#else
    temp = __float2half(__half2float(temp) + __half2float(x[i]));
#endif
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0){
        if (cacheIndex < i){
#if __CUDA_ARCH__ >= 530
    cache[cacheIndex] = __hadd(cache[cacheIndex], cache[cacheIndex+i]);
#else
    cache[cacheIndex] = __float2half(__half2float(cache[cacheIndex]) + __half2float(cache[cacheIndex+i]));
#endif
        }
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0){
        sum[blockIdx.x] = __float2half(0.0f);
        sum[blockIdx.x] = cache[0];
    }
}


__global__ void cudaH_variance_kernel(__half* x,
                                      __half* sum, 
                                      __half mean, 
                                      unsigned int size)
{
    //256 - threadsPerBlock
    __shared__ __half cache[256];

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    int cacheIndex = threadIdx.x;

    __half temp = __float2half(0.0f);
    for (unsigned int i = index; i < size; i += stride) {
        //wVariance += (weightsQ(i) - wMean)*(weightsQ(i) - wMean);
#if __CUDA_ARCH__ >= 530
    temp = __hadd(temp, __hmul(__hsub(x[i],mean),__hsub(x[i],mean)));
#else
    temp = __float2half(__half2float(temp) + (__half2float(x[i])-__half2float(mean))*(__half2float(x[i])-__half2float(mean)));
    //temp = __float2half(__half2float(temp) + __half2float(x[i]));
#endif
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0){
        if (cacheIndex < i){
#if __CUDA_ARCH__ >= 530
    cache[cacheIndex] = __hadd(cache[cacheIndex], cache[cacheIndex+i]);
#else
    cache[cacheIndex] = __float2half(__half2float(cache[cacheIndex]) + __half2float(cache[cacheIndex+i]));
#endif
        }
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0){
        sum[blockIdx.x] = __float2half(0.0f);
        sum[blockIdx.x] = cache[0];
    }
}


half_float::half N2D2::Quantizer_Frame_CUDA_Kernels::cudaH_mean(half_float::half* data, 
                                                                half_float::half* partialSum, 
                                                                const unsigned int size)
{
    return cudaH_accumulate(data, partialSum, size) / half_float::half(size);
}


half_float::half N2D2::Quantizer_Frame_CUDA_Kernels::cudaH_variance(half_float::half* data, 
                                                                    half_float::half* partialSum, 
                                                                    half_float::half mean,
                                                                    const unsigned int size)
{
    int threadsPerBlock = 256;  // Should not be changed
    int blocksPerGrid = imin(32, (size + threadsPerBlock-1) / threadsPerBlock);

    cudaH_variance_kernel<<< (size + 255) / 256, 256>>>(reinterpret_cast<__half*>(data),
                                                        reinterpret_cast<__half*>(partialSum),
                                                        reinterpret_cast<__half&>(mean),
                                                        size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

    half_float::half* hostPartSumVar = (half_float::half*)malloc(blocksPerGrid*sizeof(half_float::half));
    CHECK_CUDA_STATUS(cudaMemcpy(hostPartSumVar,
                                 partialSum,
                                 blocksPerGrid*sizeof(half_float::half),
                                 cudaMemcpyDeviceToHost));

    half_float::half sum_var = (half_float::half)0.0f;
    for (int i = 0; i<blocksPerGrid; i++){
        sum_var += hostPartSumVar[i];
    }
    half_float::half variance = sum_var/(half_float::half)(size - 1.0f);

    free(hostPartSumVar);
    return variance;
}


half_float::half N2D2::Quantizer_Frame_CUDA_Kernels::cudaH_accumulate(half_float::half* data, 
                                                                      half_float::half* partialSum, 
                                                                      const unsigned int size)
{
    int threadsPerBlock = 256;  // Should not be changed
    int blocksPerGrid = imin(32, (size + threadsPerBlock-1) / threadsPerBlock);

    cudaH_sum_kernel<<< (size + 255) / 256, 256>>>(reinterpret_cast<__half*>(data),
                                                   reinterpret_cast<__half*>(partialSum),
                                                   size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

    half_float::half* hostPartSum = (half_float::half*)malloc(blocksPerGrid*sizeof(half_float::half));
    CHECK_CUDA_STATUS(cudaMemcpy(hostPartSum,
                                 partialSum,
                                 blocksPerGrid*sizeof(half_float::half),
                                 cudaMemcpyDeviceToHost));

    half_float::half sum = (half_float::half)0.0f;
    for (int i = 0; i<blocksPerGrid; ++i){
        sum += hostPartSum[i];
    }   
    free(hostPartSum); 
    return sum;
}


float N2D2::Quantizer_Frame_CUDA_Kernels::cudaF_accumulate(float* data, 
                                                           const unsigned int size)
{
    thrust::device_ptr<float> dataPtr(data);
    return thrust::reduce(dataPtr, dataPtr+size, float(0.0));
}


double N2D2::Quantizer_Frame_CUDA_Kernels::cudaD_accumulate(double* data, 
                                                            const unsigned int size)
{
    thrust::device_ptr<double> dataPtr(data);
    return thrust::reduce(dataPtr, dataPtr+size, double(0.0));
}


__global__ void cudaH_copyData_kernel(__half* x,
                                      __half* y,
                                      unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
        __half value = x[i];
        y[i] = value;
#else
        float x_f = __half2float(x[i]);
        float value_f = x_f;
        y[i] = __float2half(value_f);
#endif
    }
}

void N2D2::Quantizer_Frame_CUDA_Kernels::cudaH_copyData(half_float::half* input, 
                                                        half_float::half* output, 
                                                        unsigned int inputSize)
{
    cudaH_copyData_kernel<<< (inputSize + 255) / 256, 256>>> (reinterpret_cast<__half*> (input), 
                                                              reinterpret_cast<__half*> (output), 
                                                              inputSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


__global__ void cudaF_copyData_kernel(float* x,
                                      float* y,
                                      unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        float value = x[i];
        y[i] = value;
    }
}

void N2D2::Quantizer_Frame_CUDA_Kernels::cudaF_copyData(float* input, 
                                                        float* output, 
                                                        unsigned int inputSize)
{
    cudaF_copyData_kernel<<< (inputSize + 255) / 256, 256>>> (input, output, inputSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


__global__ void cudaD_copyData_kernel(double* x,
                                      double* y,
                                      unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        double value = x[i];
        y[i] = value;
    }
}

void N2D2::Quantizer_Frame_CUDA_Kernels::cudaD_copyData(double* input, 
                                                        double* output, 
                                                        unsigned int inputSize)
{
    cudaD_copyData_kernel<<< (inputSize + 255) / 256, 256>>> (input, output, inputSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}