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

__global__ void cudaHTargetBiasPropagate_kernel(
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

        if (__half2float(diffInputs[i]) > 0.0f
            && __half2float(inputs[i]) > -__half2float(bias))
        {
            outputs[i] = __hadd(outputs[i], bias);
        }
    }
}

__global__ void cudaSTargetBiasPropagate_kernel(
                                         unsigned int size,
                                         const float bias,
                                         const float* inputs,
                                         const float* diffInputs,
                                         float* outputs)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        outputs[i] = inputs[i];

        if (diffInputs[i] > 0.0f && inputs[i] > -bias)
            outputs[i] += bias;
    }
}

__global__ void cudaDTargetBiasPropagate_kernel(
                                         unsigned int size,
                                         const double bias,
                                         const double* inputs,
                                         const double* diffInputs,
                                         double* outputs)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        outputs[i] = inputs[i];

        if (diffInputs[i] > 0.0 && inputs[i] > -bias)
            outputs[i] += bias;
    }
}

void N2D2::cudaHTargetBiasPropagate(const cudaDeviceProp& deviceProp,
                             const half_float::half bias,
                             const half_float::half* inputs,
                             const half_float::half* diffInputs,
                             half_float::half* outputs,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int nbChannels,
                             unsigned int batchSize)
{
    const unsigned int size = channelsHeight * channelsWidth
                                * nbChannels * batchSize;

    cudaHTargetBiasPropagate_kernel<<<(size + 255) / 256, 256>>>
        (size,
           reinterpret_cast<const __half&>(bias),
           reinterpret_cast<const __half*>(inputs),
           reinterpret_cast<const __half*>(diffInputs),
           reinterpret_cast<__half*>(outputs));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSTargetBiasPropagate(const cudaDeviceProp& deviceProp,
                             const float bias,
                             const float* inputs,
                             const float* diffInputs,
                             float* outputs,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int nbChannels,
                             unsigned int batchSize)
{
    const unsigned int size = channelsHeight * channelsWidth
                                * nbChannels * batchSize;

    cudaSTargetBiasPropagate_kernel<<<(size + 255) / 256, 256>>>
        (size,
           bias,
           inputs,
           diffInputs,
           outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDTargetBiasPropagate(const cudaDeviceProp& deviceProp,
                             const double bias,
                             const double* inputs,
                             const double* diffInputs,
                             double* outputs,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int nbChannels,
                             unsigned int batchSize)
{
    const unsigned int size = channelsHeight * channelsWidth
                                * nbChannels * batchSize;

    cudaDTargetBiasPropagate_kernel<<<(size + 255) / 256, 256>>>
        (size,
           bias,
           inputs,
           diffInputs,
           outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
