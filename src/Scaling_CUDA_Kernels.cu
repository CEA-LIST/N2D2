/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include <cassert>
#include "CudaUtils.hpp"
#include "Scaling_CUDA_Kernels.hpp"
#include "third_party/half.hpp"

using N2D2::Float_T;
using N2D2::Cuda::clamp;

template<typename T>
__device__ T saturate(T value, std::size_t quantizedNbBits, bool isOutputUnsigned) {
    assert(quantizedNbBits > 0);

    const T min = isOutputUnsigned?0:
                                  -(1ll << (quantizedNbBits - 1ll));
    const T max = isOutputUnsigned?(1ll << quantizedNbBits) - 1ll:
                                   (1ll << (quantizedNbBits - 1ll)) - 1ll;

    return clamp(value, min, max);
}

template<typename T>
__device__ T Clip(T value, Float_T clip) {
    T res = (value < T(0.0)) ? T(0.0) : (value > T(clip)) ? T(clip) : value;
    return res;
}

template<typename T>
__device__ T Scale(T value, Float_T scale) {
    T res = value*T(scale);
    return res;
}

template<typename T>
__global__ void cudaFloatingPointScaling_kernel(const T* input, T* output,
                                                std::size_t batchSize, std::size_t nbChannels,
                                                std::size_t heigth, std::size_t width,
                                                bool isClipped,
                                                Float_T* clippingFactorPerChannel,
                                                Float_T* scalingFactorPerChannel, 
                                                std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    const std::size_t startBatch = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t startI = blockIdx.y*blockDim.y + threadIdx.y;
    
    const std::size_t strideBatch = blockDim.x*gridDim.x;
    const std::size_t strideI = blockDim.y*gridDim.y;
    
    
    for (std::size_t batch = startBatch; batch < batchSize; batch += strideBatch) {
        for(std::size_t ch = 0; ch < nbChannels; ch++) {
            for (std::size_t i = startI; i < heigth*width; i += strideI) {
                const std::size_t index = batch*nbChannels*heigth*width + 
                                          ch*heigth*width +
                                          i;

                //clipping before scaling
                T res = isClipped ? Clip(input[index], clippingFactorPerChannel[ch]) 
                                    : input[index];
                res = Scale(res, scalingFactorPerChannel[ch]);
                if(quantizedNbBits > 0) {
                    res = saturate(round(res), quantizedNbBits, isOutputUnsigned);
                }
                output[index] = res;
            }
        }
    }
}

template<typename T>
__global__ void cudaFixedPointScaling_kernel(const T* input, T* output,
                                             std::size_t batchSize, std::size_t nbChannels,
                                             std::size_t heigth, std::size_t width,
                                             bool isClipped, Float_T* clippingFactorPerChannel,
                                             std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                             std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    assert(quantizedNbBits > 0);
        
    const std::size_t startBatch = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t startI = blockIdx.y*blockDim.y + threadIdx.y;
    
    const std::size_t strideBatch = blockDim.x*gridDim.x;
    const std::size_t strideI = blockDim.y*gridDim.y;
    
    
    for (std::size_t batch = startBatch; batch < batchSize; batch += strideBatch) {
        for(std::size_t ch = 0; ch < nbChannels; ch++) {
            for (std::size_t i = startI; i < heigth*width; i += strideI) {
                const std::size_t index = batch*nbChannels*heigth*width + 
                                          ch*heigth*width +
                                          i;
                
                T realInput = isClipped ? Clip(input[index], clippingFactorPerChannel[ch]) 
                                    : input[index]; 

                const long long half = (nbFractionalBits > 0)
                    ? (1ll << (nbFractionalBits - 1))
                    : 0ll;

                long long rInput = round(realInput);
                const long long res = (
                    static_cast<long long>(rInput) * scalingFactorPerChannel[ch] + half
                )  >> nbFractionalBits;
                

                output[index] = saturate(res, quantizedNbBits, isOutputUnsigned);
            }
        }
    }
}

template<typename T>
__global__ void cudaSingleShiftScaling_kernel(const T* input, T* output,
                                                std::size_t batchSize, std::size_t nbChannels,
                                                std::size_t heigth, std::size_t width,
                                                bool isClipped, Float_T* clippingFactorPerChannel,
                                                unsigned char* scalingFactorPerChannel,
                                                std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    const std::size_t startBatch = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t startI = blockIdx.y*blockDim.y + threadIdx.y;
    
    const std::size_t strideBatch = blockDim.x*gridDim.x;
    const std::size_t strideI = blockDim.y*gridDim.y;
    
    
    for (std::size_t batch = startBatch; batch < batchSize; batch += strideBatch) {
        for(std::size_t ch = 0; ch < nbChannels; ch++) {
            for (std::size_t i = startI; i < heigth*width; i += strideI) {
                const std::size_t index = batch*nbChannels*heigth*width + 
                                          ch*heigth*width +
                                          i;
                
                //TODO::add clipping here properly, nothing for now
                /*
                const long long half = (scalingFactorPerChannel[ch] > 0)
                    ? (1ll << (scalingFactorPerChannel[ch] - 1))
                    : 0ll;
                const long long res = (
                    static_cast<long long>(round(input[index])) + half
                ) >> scalingFactorPerChannel[ch];
                */
                T realInput = input[index];
                if(isClipped){
                    realInput = (realInput > T(clippingFactorPerChannel[ch])) ? T(clippingFactorPerChannel[ch]) : realInput;
                }

                const long long half = (scalingFactorPerChannel[ch] > 0)
                ? (1ll << (scalingFactorPerChannel[ch] - 1))
                : 0ll;

                long long rInput = round(realInput);

                const long long res = (
                    static_cast<long long>(rInput) + half
                ) >> scalingFactorPerChannel[ch];

                output[index] = saturate(res, quantizedNbBits, isOutputUnsigned);
            }
        }
    }
}

template<typename T>
__global__ void cudaDoubleShiftScaling_kernel(const T* input, T* output,
                                              std::size_t batchSize, std::size_t nbChannels,
                                              std::size_t heigth, std::size_t width,
                                              bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                              std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                              std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    const std::size_t startBatch = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t startI = blockIdx.y*blockDim.y + threadIdx.y;
    
    const std::size_t strideBatch = blockDim.x*gridDim.x;
    const std::size_t strideI = blockDim.y*gridDim.y;
    
    
    for (std::size_t batch = startBatch; batch < batchSize; batch += strideBatch) {
        for(std::size_t ch = 0; ch < nbChannels; ch++) {
            for (std::size_t i = startI; i < heigth*width; i += strideI) {
                const std::size_t index = batch*nbChannels*heigth*width + 
                                          ch*heigth*width +
                                          i;

                //TODO::add clipping here properly, nothing for now  
                const long long half = (scalingFactorPerChannel[ch].second > 0)
                    ? (1ll << (scalingFactorPerChannel[ch].second - 1))
                    : 0ll;
                const long long val = static_cast<long long>(round(input[index]));
                const long long res = (
                    val + (val << scalingFactorPerChannel[ch].first) +  half
                ) >> scalingFactorPerChannel[ch].second;
                
                /*
                const long long half = (scalingFactorPerChannel[ch].second > 0)
                ? (1ll << (scalingFactorPerChannel[ch].second - 1))
                : 0ll;

                long long val = round(input[index]);
                if(isClipped){
                    val = (val > clippingFactorPerChannel[ch]) ? clippingFactorPerChannel[ch] : val;
                }

                const long long res = (
                    val + (val << scalingFactorPerChannel[ch].first) +  half
                ) >> scalingFactorPerChannel[ch].second;
                */

                output[index] = saturate(res, quantizedNbBits, isOutputUnsigned);
            }
        }
    }
}





namespace N2D2 {

template<>
void cudaFloatingPointScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                const half_float::half* input, half_float::half* output,
                                                                std::size_t batchSize, std::size_t nbChannels,
                                                                std::size_t heigth, std::size_t width,
                                                                bool isClipped,
                                                                Float_T* clippingFactorPerChannel,
                                                                Float_T* scalingFactorPerChannel,
                                                                std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    throw std::runtime_error("Floating-point scaling cell doesn't support half-floats.");
}


template<typename T>
void cudaFloatingPointScaling_propagate(const cudaDeviceProp& deviceProp,
                                              const T* input, T* output,
                                              std::size_t batchSize, std::size_t nbChannels,
                                              std::size_t heigth, std::size_t width,
                                              bool isClipped,
                                              Float_T* clippingFactorPerChannel,
                                              Float_T* scalingFactorPerChannel,
                                              std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    // TODO Optimize dimensions based on the size of the batch and cell
    const dim3 threadsPerBlock = dim3(deviceProp.maxThreadsPerBlock/deviceProp.warpSize, 
                            deviceProp.warpSize);
    const dim3 blocksPerGrid = dim3(16, deviceProp.multiProcessorCount);

    cudaFloatingPointScaling_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, 
                                                                        batchSize, nbChannels, 
                                                                        heigth, width, 
                                                                        isClipped,
                                                                        clippingFactorPerChannel,
                                                                        scalingFactorPerChannel,
                                                                        quantizedNbBits, isOutputUnsigned);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}






template<>
void cudaFixedPointScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                             const half_float::half* input, half_float::half* output,
                                                             std::size_t batchSize, std::size_t nbChannels,
                                                             std::size_t heigth, std::size_t width,
                                                             bool isClipped, Float_T* clippingFactorPerChannel,
                                                             std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                             std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    throw std::runtime_error("Fixed-point scaling cell doesn't support half-floats.");
}

template<typename T>
void cudaFixedPointScaling_propagate(const cudaDeviceProp& deviceProp,
                                           const T* input, T* output,
                                           std::size_t batchSize, std::size_t nbChannels,
                                           std::size_t heigth, std::size_t width,
                                           bool isClipped, Float_T* clippingFactorPerChannel,
                                           std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                           std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    // TODO Optimize dimensions based on the size of the batch and cell
    const dim3 threadsPerBlock = dim3(deviceProp.maxThreadsPerBlock/deviceProp.warpSize, 
                            deviceProp.warpSize);
    const dim3 blocksPerGrid = dim3(16, deviceProp.multiProcessorCount);

    cudaFixedPointScaling_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, 
                                                                     batchSize, nbChannels, 
                                                                     heigth, width, 
                                                                     isClipped, clippingFactorPerChannel,
                                                                     scalingFactorPerChannel, nbFractionalBits,
                                                                     quantizedNbBits, isOutputUnsigned);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}







template<>
void cudaSingleShiftScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                              const half_float::half* input, half_float::half* output,
                                                              std::size_t batchSize, std::size_t nbChannels,
                                                              std::size_t heigth, std::size_t width,
                                                              bool isClipped, Float_T* clippingFactorPerChannel,
                                                              unsigned char* scalingFactorPerChannel,
                                                              std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    throw std::runtime_error("Single-shift scaling cell doesn't support half-floats.");
}

template<typename T>
void cudaSingleShiftScaling_propagate(const cudaDeviceProp& deviceProp,
                                            const T* input, T* output,
                                            std::size_t batchSize, std::size_t nbChannels,
                                            std::size_t heigth, std::size_t width,
                                            bool isClipped, Float_T* clippingFactorPerChannel,
                                            unsigned char* scalingFactorPerChannel,
                                            std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    // TODO Optimize dimensions based on the size of the batch and cell
    const dim3 threadsPerBlock = dim3(deviceProp.maxThreadsPerBlock/deviceProp.warpSize, 
                            deviceProp.warpSize);
    const dim3 blocksPerGrid = dim3(16, deviceProp.multiProcessorCount);

    cudaSingleShiftScaling_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, 
                                                                      batchSize, nbChannels, 
                                                                      heigth, width, 
                                                                      isClipped, clippingFactorPerChannel,
                                                                      scalingFactorPerChannel,
                                                                      quantizedNbBits, isOutputUnsigned);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}






template<>
void cudaDoubleShiftScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                              const half_float::half* input, half_float::half* output,
                                                              std::size_t batchSize, std::size_t nbChannels,
                                                              std::size_t heigth, std::size_t width,
                                                              bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                                              std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                                              std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    throw std::runtime_error("Double-shift scaling cell doesn't support half-floats.");
}

template<typename T>
void cudaDoubleShiftScaling_propagate(const cudaDeviceProp& deviceProp,
                                            const T* input, T* output,
                                            std::size_t batchSize, std::size_t nbChannels,
                                            std::size_t heigth, std::size_t width,
                                            bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                            std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                            std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    // TODO Optimize dimensions based on the size of the batch and cell
    const dim3 threadsPerBlock = dim3(deviceProp.maxThreadsPerBlock/deviceProp.warpSize, 
                            deviceProp.warpSize);
    const dim3 blocksPerGrid = dim3(16, deviceProp.multiProcessorCount);

    cudaDoubleShiftScaling_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, 
                                                                      batchSize, nbChannels, 
                                                                      heigth, width, 
                                                                      isClipped, clippingFactorPerChannel,
                                                                      scalingFactorPerChannel,
                                                                      quantizedNbBits, isOutputUnsigned);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}






template void cudaFloatingPointScaling_propagate<float>(const cudaDeviceProp& deviceProp,
                                                              const float* input, float* output,
                                                              std::size_t batchSize, std::size_t nbChannels,
                                                              std::size_t heigth, std::size_t width,
                                                              bool isClipped, Float_T* clippingFactorPerChannel,
                                                              Float_T* scalingFactorPerChannel,
                                                              std::size_t quantizedNbBits, bool isOutputUnsigned);

template void cudaFloatingPointScaling_propagate<double>(const cudaDeviceProp& deviceProp,
                                                               const double* input, double* output,
                                                               std::size_t batchSize, std::size_t nbChannels,
                                                               std::size_t heigth, std::size_t width,
                                                               bool isClipped, Float_T* clippingFactorPerChannel,
                                                               Float_T* scalingFactorPerChannel,
                                                               std::size_t quantizedNbBits, bool isOutputUnsigned);

template void cudaFloatingPointScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                         const half_float::half* input, half_float::half* output,
                                                                         std::size_t batchSize, std::size_t nbChannels,
                                                                         std::size_t heigth, std::size_t width,
                                                                         bool isClipped, Float_T* clippingFactorPerChannel,
                                                                         Float_T* scalingFactorPerChannel,
                                                                         std::size_t quantizedNbBits, bool isOutputUnsigned);


template void cudaFixedPointScaling_propagate<float>(const cudaDeviceProp& deviceProp,
                                                           const float* input, float* output,
                                                           std::size_t batchSize, std::size_t nbChannels,
                                                           std::size_t heigth, std::size_t width,
                                                           bool isClipped, Float_T* clippingFactorPerChannel,
                                                           std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                           std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaFixedPointScaling_propagate<double>(const cudaDeviceProp& deviceProp,
                                                            const double* input, double* output,
                                                            std::size_t batchSize, std::size_t nbChannels,
                                                            std::size_t heigth, std::size_t width,
                                                            bool isClipped, Float_T* clippingFactorPerChannel,
                                                            std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                            std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaFixedPointScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                      const half_float::half* input, half_float::half* output,
                                                                      std::size_t batchSize, std::size_t nbChannels,
                                                                      std::size_t heigth, std::size_t width,
                                                                      bool isClipped, Float_T* clippingFactorPerChannel,
                                                                      std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                                      std::size_t quantizedNbBits, bool isOutputUnsigned);


template void cudaSingleShiftScaling_propagate<float>(const cudaDeviceProp& deviceProp,
                                                            const float* input, float* output,
                                                            std::size_t batchSize, std::size_t nbChannels,
                                                            std::size_t heigth, std::size_t width,
                                                            bool isClipped, Float_T* clippingFactorPerChannel,
                                                            unsigned char* scalingFactorPerChannel,
                                                            std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaSingleShiftScaling_propagate<double>(const cudaDeviceProp& deviceProp,
                                                             const double* input, double* output,
                                                             std::size_t batchSize, std::size_t nbChannels,
                                                             std::size_t heigth, std::size_t width,
                                                             bool isClipped, Float_T* clippingFactorPerChannel,
                                                             unsigned char* scalingFactorPerChannel,
                                                             std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaSingleShiftScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                       const half_float::half* input, half_float::half* output,
                                                                       std::size_t batchSize, std::size_t nbChannels,
                                                                       std::size_t heigth, std::size_t width,
                                                                       bool isClipped, Float_T* clippingFactorPerChannel,
                                                                       unsigned char* scalingFactorPerChannel,
                                                                       std::size_t quantizedNbBits, bool isOutputUnsigned);


template void cudaDoubleShiftScaling_propagate<float>(const cudaDeviceProp& deviceProp,
                                                            const float* input, float* output,
                                                            std::size_t batchSize, std::size_t nbChannels,
                                                            std::size_t heigth, std::size_t width,
                                                            bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                                            std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                                            std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaDoubleShiftScaling_propagate<double>(const cudaDeviceProp& deviceProp,
                                                             const double* input, double* output,
                                                             std::size_t batchSize, std::size_t nbChannels,
                                                             std::size_t heigth, std::size_t width,
                                                             bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                                             std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                                             std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaDoubleShiftScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                       const half_float::half* input, half_float::half* output,
                                                                       std::size_t batchSize, std::size_t nbChannels,
                                                                       std::size_t heigth, std::size_t width,
                                                                       bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                                                       std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                                                       std::size_t quantizedNbBits, bool isOutputUnsigned);
}