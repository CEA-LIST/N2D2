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
                                                std::size_t height, std::size_t width,
                                                bool isClipped,
                                                Float_T* clippingFactorPerChannel,
                                                Float_T* scalingFactorPerChannel, 
                                                std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    const unsigned int batchOffset = blockIdx.z * nbChannels * height * width;

    for (unsigned int z = blockIdx.x; z < nbChannels; z += gridDim.x) {
        for (unsigned int y = threadIdx.y; y < height; y += blockDim.y) {
            for (unsigned int x = threadIdx.x; x < width; x += blockDim.x) {
                const unsigned int idx
                    = x + width * (y + height * z) + batchOffset;

                T res = isClipped ? Clip(input[idx], clippingFactorPerChannel[z])
                                  : input[idx];
                res = Scale(res, scalingFactorPerChannel[z]);
                if(quantizedNbBits > 0) {
                    res = saturate(round(res), quantizedNbBits, isOutputUnsigned);
                }
                output[idx] = res;
            }
        }
    }

}

template<typename T>
__global__ void cudaFixedPointScaling_kernel(const T* input, T* output,
                                             std::size_t batchSize, std::size_t nbChannels,
                                             std::size_t height, std::size_t width,
                                             bool isClipped, Float_T* clippingFactorPerChannel,
                                             std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                             std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    assert(quantizedNbBits > 0);

    const unsigned int batchOffset = blockIdx.z * nbChannels * height * width;

    for (unsigned int z = blockIdx.x; z < nbChannels; z += gridDim.x) {
        for (unsigned int y = threadIdx.y; y < height; y += blockDim.y) {
            for (unsigned int x = threadIdx.x; x < width; x += blockDim.x) {
                const unsigned int index
                    = x + width * (y + height * z) + batchOffset;
                
                T realInput = isClipped ? Clip(input[index], clippingFactorPerChannel[z]) 
                                    : input[index]; 

                const long long half = (nbFractionalBits > 0)
                    ? (1ll << (nbFractionalBits - 1))
                    : 0ll;

                long long rInput = round(realInput);
                const long long res = (
                    static_cast<long long>(rInput) * scalingFactorPerChannel[z] + half
                )  >> nbFractionalBits;
                

                output[index] = saturate(res, quantizedNbBits, isOutputUnsigned);
            }
        }
    }
}

template<typename T>
__global__ void cudaSingleShiftScaling_kernel(const T* input, T* output,
                                                std::size_t batchSize, std::size_t nbChannels,
                                                std::size_t height, std::size_t width,
                                                bool isClipped, Float_T* clippingFactorPerChannel,
                                                unsigned char* scalingFactorPerChannel,
                                                std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    const unsigned int batchOffset = blockIdx.z * nbChannels * height * width;

    for (unsigned int z = blockIdx.x; z < nbChannels; z += gridDim.x) {
        for (unsigned int y = threadIdx.y; y < height; y += blockDim.y) {
            for (unsigned int x = threadIdx.x; x < width; x += blockDim.x) {
                const unsigned int index
                    = x + width * (y + height * z) + batchOffset;
                
                T realInput = input[index];
                if(isClipped){
                    realInput = (realInput > T(clippingFactorPerChannel[z])) ? T(clippingFactorPerChannel[z]) : realInput;
                }

                const long long half = (scalingFactorPerChannel[z] > 0)
                ? (1ll << (scalingFactorPerChannel[z] - 1))
                : 0ll;

                long long rInput = round(realInput);

                const long long res = (
                    static_cast<long long>(rInput) + half
                ) >> scalingFactorPerChannel[z];

                output[index] = saturate(res, quantizedNbBits, isOutputUnsigned);
            }
        }
    }
}

//TODO adapt for QAT with clipping and scaling factors
template<typename T>
__global__ void cudaDoubleShiftScaling_kernel(const T* input, T* output,
                                              std::size_t batchSize, std::size_t nbChannels,
                                              std::size_t height, std::size_t width,
                                              bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                              std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                              std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    const unsigned int batchOffset = blockIdx.z * nbChannels * height * width;

    for (unsigned int z = blockIdx.x; z < nbChannels; z += gridDim.x) {
        for (unsigned int y = threadIdx.y; y < height; y += blockDim.y) {
            for (unsigned int x = threadIdx.x; x < width; x += blockDim.x) {
                const unsigned int idx
                    = x + width * (y + height * z) + batchOffset;

                const long long half = (scalingFactorPerChannel[z].second > 0)
                    ? (1ll << (scalingFactorPerChannel[z].second - 1))
                    : 0ll;
                const long long val = static_cast<long long>(round(input[idx]));
                const long long res = (
                    val + (val << scalingFactorPerChannel[z].first) +  half
                ) >> scalingFactorPerChannel[z].second;


                output[idx] = saturate(res, quantizedNbBits, isOutputUnsigned);
            }
        }
    }
}





namespace N2D2 {

template<>
void cudaFloatingPointScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                const half_float::half* input, half_float::half* output,
                                                                std::size_t batchSize, std::size_t nbChannels,
                                                                std::size_t height, std::size_t width,
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
                                              std::size_t height, std::size_t width,
                                              bool isClipped,
                                              Float_T* clippingFactorPerChannel,
                                              Float_T* scalingFactorPerChannel,
                                              std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    // Dynamic determination of kernel attributes may lead to bad behaviours
    // https://stackoverflow.com/questions/26201172/cuda-too-many-resources-requested-for-launch
    // 
    // "interestingly, I found with a particular kernel (quite large, with lots of nested calls) 
    // I could launch with 512 threads in debug, but when compiled for release, that failed, 
    // and it would only work with 256. A release build had heavier register usage it seems"

    // const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int maxSize = 256U;

    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (width * height < maxSize)
                                       ? width * height
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)width);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {(unsigned int)nbChannels, 1, (unsigned int)batchSize};
    const dim3 threadsPerBlock = {groupWidth, groupSize / groupWidth, 1};

    cudaFloatingPointScaling_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, 
                                                                        batchSize, nbChannels, 
                                                                        height, width, 
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
                                                             std::size_t height, std::size_t width,
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
                                           std::size_t height, std::size_t width,
                                           bool isClipped, Float_T* clippingFactorPerChannel,
                                           std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                           std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    // Dynamic determination of kernel attributes may lead to bad behaviours
    // https://stackoverflow.com/questions/26201172/cuda-too-many-resources-requested-for-launch
    // 
    // "interestingly, I found with a particular kernel (quite large, with lots of nested calls) 
    // I could launch with 512 threads in debug, but when compiled for release, that failed, 
    // and it would only work with 256. A release build had heavier register usage it seems"

    // const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int maxSize = 256U;

    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (width * height < maxSize)
                                       ? width * height
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)width);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {(unsigned int)nbChannels, 1, (unsigned int)batchSize};
    const dim3 threadsPerBlock = {groupWidth, groupSize / groupWidth, 1};

    cudaFixedPointScaling_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, 
                                                                     batchSize, nbChannels, 
                                                                     height, width, 
                                                                     isClipped, clippingFactorPerChannel,
                                                                     scalingFactorPerChannel, nbFractionalBits,
                                                                     quantizedNbBits, isOutputUnsigned);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}







template<>
void cudaSingleShiftScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                              const half_float::half* input, half_float::half* output,
                                                              std::size_t batchSize, std::size_t nbChannels,
                                                              std::size_t height, std::size_t width,
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
                                            std::size_t height, std::size_t width,
                                            bool isClipped, Float_T* clippingFactorPerChannel,
                                            unsigned char* scalingFactorPerChannel,
                                            std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    // Dynamic determination of kernel attributes may lead to bad behaviours
    // https://stackoverflow.com/questions/26201172/cuda-too-many-resources-requested-for-launch
    // 
    // "interestingly, I found with a particular kernel (quite large, with lots of nested calls) 
    // I could launch with 512 threads in debug, but when compiled for release, that failed, 
    // and it would only work with 256. A release build had heavier register usage it seems"

    //const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int maxSize = 256U;

    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (width * height < maxSize)
                                       ? width * height
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)width);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {(unsigned int)nbChannels, 1, (unsigned int)batchSize};
    const dim3 threadsPerBlock = {groupWidth, groupSize / groupWidth, 1};

    cudaSingleShiftScaling_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, 
                                                                      batchSize, nbChannels, 
                                                                      height, width, 
                                                                      isClipped, clippingFactorPerChannel,
                                                                      scalingFactorPerChannel,
                                                                      quantizedNbBits, isOutputUnsigned);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}






template<>
void cudaDoubleShiftScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                              const half_float::half* input, half_float::half* output,
                                                              std::size_t batchSize, std::size_t nbChannels,
                                                              std::size_t height, std::size_t width,
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
                                            std::size_t height, std::size_t width,
                                            bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                            std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                            std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    // Dynamic determination of kernel attributes may lead to bad behaviours
    // https://stackoverflow.com/questions/26201172/cuda-too-many-resources-requested-for-launch
    // 
    // "interestingly, I found with a particular kernel (quite large, with lots of nested calls) 
    // I could launch with 512 threads in debug, but when compiled for release, that failed, 
    // and it would only work with 256. A release build had heavier register usage it seems"

    // const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int maxSize = 256U;

    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (width * height < maxSize)
                                       ? width * height
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)width);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {(unsigned int)nbChannels, 1, (unsigned int)batchSize};
    const dim3 threadsPerBlock = {groupWidth, groupSize / groupWidth, 1};

    cudaDoubleShiftScaling_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, 
                                                                      batchSize, nbChannels, 
                                                                      height, width, 
                                                                      isClipped, clippingFactorPerChannel,
                                                                      scalingFactorPerChannel,
                                                                      quantizedNbBits, isOutputUnsigned);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}






template void cudaFloatingPointScaling_propagate<float>(const cudaDeviceProp& deviceProp,
                                                              const float* input, float* output,
                                                              std::size_t batchSize, std::size_t nbChannels,
                                                              std::size_t height, std::size_t width,
                                                              bool isClipped, Float_T* clippingFactorPerChannel,
                                                              Float_T* scalingFactorPerChannel,
                                                              std::size_t quantizedNbBits, bool isOutputUnsigned);

template void cudaFloatingPointScaling_propagate<double>(const cudaDeviceProp& deviceProp,
                                                               const double* input, double* output,
                                                               std::size_t batchSize, std::size_t nbChannels,
                                                               std::size_t height, std::size_t width,
                                                               bool isClipped, Float_T* clippingFactorPerChannel,
                                                               Float_T* scalingFactorPerChannel,
                                                               std::size_t quantizedNbBits, bool isOutputUnsigned);

template void cudaFloatingPointScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                         const half_float::half* input, half_float::half* output,
                                                                         std::size_t batchSize, std::size_t nbChannels,
                                                                         std::size_t height, std::size_t width,
                                                                         bool isClipped, Float_T* clippingFactorPerChannel,
                                                                         Float_T* scalingFactorPerChannel,
                                                                         std::size_t quantizedNbBits, bool isOutputUnsigned);


template void cudaFixedPointScaling_propagate<float>(const cudaDeviceProp& deviceProp,
                                                           const float* input, float* output,
                                                           std::size_t batchSize, std::size_t nbChannels,
                                                           std::size_t height, std::size_t width,
                                                           bool isClipped, Float_T* clippingFactorPerChannel,
                                                           std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                           std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaFixedPointScaling_propagate<double>(const cudaDeviceProp& deviceProp,
                                                            const double* input, double* output,
                                                            std::size_t batchSize, std::size_t nbChannels,
                                                            std::size_t height, std::size_t width,
                                                            bool isClipped, Float_T* clippingFactorPerChannel,
                                                            std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                            std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaFixedPointScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                      const half_float::half* input, half_float::half* output,
                                                                      std::size_t batchSize, std::size_t nbChannels,
                                                                      std::size_t height, std::size_t width,
                                                                      bool isClipped, Float_T* clippingFactorPerChannel,
                                                                      std::int32_t* scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                                      std::size_t quantizedNbBits, bool isOutputUnsigned);


template void cudaSingleShiftScaling_propagate<float>(const cudaDeviceProp& deviceProp,
                                                            const float* input, float* output,
                                                            std::size_t batchSize, std::size_t nbChannels,
                                                            std::size_t height, std::size_t width,
                                                            bool isClipped, Float_T* clippingFactorPerChannel,
                                                            unsigned char* scalingFactorPerChannel,
                                                            std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaSingleShiftScaling_propagate<double>(const cudaDeviceProp& deviceProp,
                                                             const double* input, double* output,
                                                             std::size_t batchSize, std::size_t nbChannels,
                                                             std::size_t height, std::size_t width,
                                                             bool isClipped, Float_T* clippingFactorPerChannel,
                                                             unsigned char* scalingFactorPerChannel,
                                                             std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaSingleShiftScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                       const half_float::half* input, half_float::half* output,
                                                                       std::size_t batchSize, std::size_t nbChannels,
                                                                       std::size_t height, std::size_t width,
                                                                       bool isClipped, Float_T* clippingFactorPerChannel,
                                                                       unsigned char* scalingFactorPerChannel,
                                                                       std::size_t quantizedNbBits, bool isOutputUnsigned);


template void cudaDoubleShiftScaling_propagate<float>(const cudaDeviceProp& deviceProp,
                                                            const float* input, float* output,
                                                            std::size_t batchSize, std::size_t nbChannels,
                                                            std::size_t height, std::size_t width,
                                                            bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                                            std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                                            std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaDoubleShiftScaling_propagate<double>(const cudaDeviceProp& deviceProp,
                                                             const double* input, double* output,
                                                             std::size_t batchSize, std::size_t nbChannels,
                                                             std::size_t height, std::size_t width,
                                                             bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                                             std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                                             std::size_t quantizedNbBits, bool isOutputUnsigned);
template void cudaDoubleShiftScaling_propagate<half_float::half>(const cudaDeviceProp& deviceProp,
                                                                       const half_float::half* input, half_float::half* output,
                                                                       std::size_t batchSize, std::size_t nbChannels,
                                                                       std::size_t height, std::size_t width,
                                                                       bool isClipped, std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                                                       std::pair<unsigned char, unsigned char>* scalingFactorPerChannel,
                                                                       std::size_t quantizedNbBits, bool isOutputUnsigned);
}