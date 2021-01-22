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

#include "Cell/TargetBiasCell_Frame_CUDA_Kernels.hpp"

template <class T>
__global__ void cudaTargetBiasPropagate_kernel(
                                         unsigned int size,
                                         const T bias,
                                         const T* inputs,
                                         const T* diffInputs,
                                         T* outputs)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        outputs[i] = inputs[i];

        if (diffInputs[i] > 0.0f && inputs[i] > -bias)
            outputs[i] += bias;
    }
}

template <>
__global__ void cudaTargetBiasPropagate_kernel<__half>(
                                         unsigned int size,
                                         const __half bias,
                                         const __half* inputs,
                                         const __half* diffInputs,
                                         __half* outputs)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        outputs[i] = inputs[i];

#if __CUDA_ARCH__ >= 530
        if (__hgt(diffInputs[i], __float2half(0.0f))
            && __hgt(inputs[i], __hneg(bias)))
        {
            outputs[i] = __hadd(outputs[i], bias);
        }
#else
        if (__half2float(diffInputs[i]) > 0.0f
            && __half2float(inputs[i]) > -__half2float(bias))
        {
            outputs[i] = __float2half(__half2float(outputs[i])
                + __half2float(bias));
        }
#endif
    }
}


namespace N2D2 {

template <class T>
void cudaTargetBiasPropagate(const cudaDeviceProp& deviceProp,
                             const T bias,
                             const T* inputs,
                             const T* diffInputs,
                             T* outputs,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int nbChannels,
                             unsigned int batchSize)
{
    const unsigned int size = channelsHeight * channelsWidth
                                * nbChannels * batchSize;

    cudaTargetBiasPropagate_kernel<<<(size + 255) / 256, 256>>>
        (size,
           reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(bias),
           reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(inputs),
           reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(diffInputs),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(outputs));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


template void cudaTargetBiasPropagate(const cudaDeviceProp& deviceProp,
                             const half_float::half bias,
                             const half_float::half* inputs,
                             const half_float::half* diffInputs,
                             half_float::half* outputs,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int nbChannels,
                             unsigned int batchSize);
template void cudaTargetBiasPropagate(const cudaDeviceProp& deviceProp,
                             const float bias,
                             const float* inputs,
                             const float* diffInputs,
                             float* outputs,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int nbChannels,
                             unsigned int batchSize);
template void cudaTargetBiasPropagate(const cudaDeviceProp& deviceProp,
                             const double bias,
                             const double* inputs,
                             const double* diffInputs,
                             double* outputs,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int nbChannels,
                             unsigned int batchSize);

}
