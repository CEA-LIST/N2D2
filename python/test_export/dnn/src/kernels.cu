/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "common_cuda.hpp"
#include "kernels.hpp"
#include "../../include/utils.h"
#include "../../include/params.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

__device__ __inline__ DATA_T sat32(SUM_T x) 
{
    return (DATA_T)((x > DATA_T_MAX) ? DATA_T_MAX
                                     : (x < DATA_T_MIN) ? DATA_T_MIN : x);
}

__device__ __inline__ UDATA_T usat32(SUM_T x)
{
    return (UDATA_T)((x > UDATA_T_MAX) ? UDATA_T_MAX 
                                       : (x < 0) ? 0 : x);
}

__device__ __inline__ int clamp(int x, int min, int max)
{
    return (x < min) ? min : (x > max) ? max : x;
}

__device__ __inline__ DATA_T sat(SUM_T weightedSum, ActivationFunction_T func, int shift)
{
    #if NB_BITS >= 0
        if (shift > 0)
            weightedSum >>= shift;
        else if (shift < 0)
            weightedSum <<= (-shift);
    #endif


    switch (func) {
    case Tanh:
    case TanhLeCun:
#if NB_BITS < 0
        return tanh(weightedSum);
#endif

    case Saturation:
        return sat32(weightedSum);

    case Logistic:
    case LogisticWithLoss:
#if NB_BITS < 0
        return 1.0/(1.0 + exp(-weightedSum));
#else
        return sat32(weightedSum);
#endif

    case Rectifier:
#if NB_BITS < 0
    return MAX((SUM_T)0, weightedSum);
#else
    return usat32(max((SUM_T)0, weightedSum));
#endif

    case Linear:
#if NB_BITS < 0
        return weightedSum;
#else
        return sat32(weightedSum);
#endif

    default:
        //printf("Unsupported activation function in sat()\n");
        return 0;
    }
}

__device__ __inline__ DATA_T usat(SUM_T weightedSum, ActivationFunction_T func, int shift)
{
#if NB_BITS >= 0
    if (shift > 0)
        weightedSum >>= shift;
    else if (shift < 0)
        weightedSum <<= (-shift);
#endif

    switch (func) {
    case Tanh:
    case TanhLeCun:
#if NB_BITS < 0
        return tanh(weightedSum);
#endif

    case Saturation:
        return sat32(weightedSum);

    case Logistic:
    case LogisticWithLoss:
#if NB_BITS < 0
        return 1.0/(1.0 + exp(-weightedSum));
#else
        return sat32(weightedSum);
#endif

    case Rectifier:
#if NB_BITS < 0
        return max((SUM_T)0, weightedSum);
#else
        return usat32(max((SUM_T)0, weightedSum));
#endif

    case Linear:
#if NB_BITS < 0
        return weightedSum;
#else
        return sat32(weightedSum);
#endif

    default:
        //printf("Unsupported activation function in usat()\n");
        return 0;
    }
}

__global__ void convcell_upropagate_kernel(unsigned int nbChannels,
                                           unsigned int channelsHeight,
                                           unsigned int channelsWidth,
                                           int paddingY,
                                           int paddingX,
                                           unsigned int strideY,
                                           unsigned int strideX,
                                           unsigned int subSampleY,
                                           unsigned int subSampleX,
                                           const UDATA_T* inputs,
                                           unsigned int oySize,
                                           unsigned int oxSize,
                                           unsigned int nbOutputs_,
                                           unsigned int outputsHeight,
                                           unsigned int outputsWidth,
                                           unsigned int nbOutputs,
                                           unsigned int outputOffset,
                                           DATA_T* outputs,
                                           unsigned int kernelHeight,
                                           unsigned int kernelWidth,
                                           const BDATA_T* bias,
                                           const WDATA_T* weights,
                                           ActivationFunction_T func,
                                           int shift)
{
    // Fallback
    if (subSampleY != 1 || subSampleX != 1) {
        // printf("convcell_upropagate(): subsampling not implemented\n");
        return;
    }
    const unsigned int batchInputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels * channelsHeight
          * channelsWidth; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z)
          * nbOutputs_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x) {
                const unsigned int sxMin
                    = (unsigned int)max((int)paddingX - (int)(ox * strideX), 0);
                const unsigned int syMin
                    = (unsigned int)max((int)paddingY - (int)(oy * strideY), 0);
                const unsigned int sxMax = (unsigned int)clamp(
                    (int)(channelsWidth + paddingX) - (int)(ox * strideX),
                    0,
                    (int)kernelWidth);
                const unsigned int syMax = (unsigned int)clamp(
                    (int)(channelsHeight + paddingY) - (int)(oy * strideY),
                    0,
                    (int)kernelHeight);

                const unsigned int outputsIdx = ox + (oy + output * oySize)
                                                     * oxSize;

                SUM_T weightedSum = bias[output];

                for (unsigned int ch = 0; ch != nbChannels; ++ch) {
                    const unsigned int weightsOffset
                        = (ch + output * nbChannels) * kernelWidth
                          * kernelHeight;

                    for (unsigned int sy = syMin; sy != syMax; ++sy) {
                        const int ix = (int)(ox * strideX + 0) - (int)paddingX;
                        const int iy = (int)(oy * strideY + sy) - (int)paddingY;
                        const unsigned int weightsIdx = sy * kernelWidth
                                                        + weightsOffset;
                        const unsigned int inputsIdx
                            = ix + (iy + ch * channelsHeight) * channelsWidth;

                        unsigned int sx = 0;
                        for (; sx < kernelWidth; ++sx) {
                            if (sx >= sxMin && sx < sxMax)
                                weightedSum += weights[sx + weightsIdx]
                                               * inputs[sx + inputsIdx
                                                        + batchInputOffset];
                        }
                    }
                }

                outputs[batchOutputOffset + outputOffset + outputsIdx]
                    = usat(weightedSum, func, shift);
            }
        }
    }
}

__global__ void convcell_propagate_kernel(unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth,
                                          int paddingY,
                                          int paddingX,
                                          unsigned int strideY,
                                          unsigned int strideX,
                                          unsigned int subSampleY,
                                          unsigned int subSampleX,
                                          const DATA_T* inputs,
                                          unsigned int oySize,
                                          unsigned int oxSize,
                                          unsigned int nbOutputs_,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int nbOutputs,
                                          unsigned int outputOffset,
                                          DATA_T* outputs,
                                          unsigned int kernelHeight,
                                          unsigned int kernelWidth,
                                          const BDATA_T* bias,
                                          const WDATA_T* weights,
                                          ActivationFunction_T func,
                                          int shift)
{

    // Fallback
    if (subSampleY != 1 || subSampleX != 1) {
        // printf("convcell_upropagate(): subsampling not implemented\n");
        return;
    }
    const unsigned int batchInputOffset = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels * channelsHeight * channelsWidth; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset = (blockIdx.z * blockDim.z + threadIdx.z) * nbOutputs_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs; output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight; oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth; ox += blockDim.x) {

                const unsigned int sxMin = (unsigned int)max((int)paddingX - (int)(ox * strideX), 0);
                const unsigned int syMin = (unsigned int)max((int)paddingY - (int)(oy * strideY), 0);

                const unsigned int sxMax = (unsigned int)clamp((int)(channelsWidth + paddingX) - (int)(ox * strideX), 0,  (int)kernelWidth);
                const unsigned int syMax = (unsigned int)clamp((int)(channelsHeight + paddingY) - (int)(oy * strideY), 0, (int)kernelHeight);

                const unsigned int outputsIdx = ox + (oy + output * oySize) * oxSize;

                SUM_T weightedSum = bias[output];

                for (unsigned int ch = 0; ch != nbChannels; ++ch) {
                    const unsigned int weightsOffset = (ch + output*nbChannels)*kernelWidth*kernelHeight;

                    for (unsigned int sy = syMin; sy != syMax; ++sy) {
                        const int ix = (int)(ox*strideX + 0) - (int)paddingX;
                        const int iy = (int)(oy*strideY + sy) - (int)paddingY;
                        const unsigned int weightsIdx = sy*kernelWidth + weightsOffset;
                        const unsigned int inputsIdx = ix + (iy + ch * channelsHeight)*channelsWidth;

                        unsigned int sx = 0;

                        for (; sx < kernelWidth; ++sx) {
                            if (sx >= sxMin && sx < sxMax)
                                weightedSum += weights[sx + weightsIdx]*inputs[sx + inputsIdx + batchInputOffset];
                        }
                    }
                }

                outputs[batchOutputOffset + outputOffset + outputsIdx] = sat(weightedSum, func, shift);
            }
        }
    }
}

__global__ void poolcell_upropagate_kernel(unsigned int nbChannels,
                                           unsigned int channelsHeight,
                                           unsigned int channelsWidth,
                                           unsigned int strideY,
                                           unsigned int strideX,
                                           const UDATA_T* inputs,
                                           unsigned int nbOutputs_,
                                           unsigned int outputsHeight,
                                           unsigned int outputsWidth,
                                           unsigned int nbOutputs,
                                           unsigned int outputOffset,
                                           DATA_T* outputs,
                                           unsigned int poolHeight,
                                           unsigned int poolWidth,
                                           const char* mapping,
                                           Pooling_T pooling,
                                           ActivationFunction_T func,
                                           int shift)
{
#if NB_BITS < 0
    if(func != Linear)
        return;
#endif
    const unsigned int batchInputOffset = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels * channelsHeight * channelsWidth; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset = (blockIdx.z * blockDim.z + threadIdx.z) * nbOutputs_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x) {

                const unsigned int sxMax = min(channelsWidth - ox*strideX, (unsigned int)poolWidth);
                const unsigned int syMax = min(channelsHeight - oy*strideY, (unsigned int)poolHeight);

                const unsigned int outputsIdx = ox + (oy + output * outputsHeight) * outputsWidth;

                if (pooling == Max) {
                    UDATA_T poolValue = 0;

                    for (unsigned int ch = 0; ch != nbChannels; ++ch) {

                        if (!mapping[ch + output * nbChannels])
                             continue;

                        for (unsigned int sy = 0; sy != syMax; ++sy) {
                            const unsigned int ix = ox*strideX + 0;
                            const unsigned int iy = oy*strideY + sy;
                            const unsigned int inputsIdx = ix + (iy + output*channelsHeight)*channelsWidth;

                            unsigned int sx = 0;

                            for (; sx < sxMax; ++sx) {
                                if (inputs[sx + inputsIdx + batchInputOffset]
                                    > poolValue)
                                    poolValue
                                        = inputs[sx + inputsIdx + batchInputOffset];
                            }
                        }
                    }
#if NB_BITS < 0
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = usat(poolValue, func, shift);
#else
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = poolValue;
#endif
                }
                else if (pooling == Average) {
                    unsigned int nbMapChan = 0;
                    SUM_T sum = 0;
                    for (unsigned int ch = 0; ch != nbChannels; ++ch) {
                        if (!mapping[ch + output*nbChannels])
                            continue;

                        for (unsigned int sy = 0; sy != syMax; ++sy) {
                            const unsigned int ix = ox*strideX + 0;
                            const unsigned int iy = oy*strideY + sy;
                            const unsigned int inputsIdx = ix + (iy + output*channelsHeight)*channelsWidth;

                            for (unsigned int sx = 0; sx < sxMax; ++sx)
                                    sum += inputs[sx + inputsIdx + batchInputOffset];
                        }

                        ++nbMapChan;
                    }

                    sum /= (poolHeight*poolWidth*nbMapChan);
#if NB_BITS < 0
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = usat(sum, func, shift);
#else
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = sum;
#endif
                }
            }
        }
     }
}
__global__ void poolcell_upropagate_unitmap_kernel(unsigned int nbChannels,
                                                   unsigned int channelsHeight,
                                                   unsigned int channelsWidth,
                                                   unsigned int strideY,
                                                   unsigned int strideX,
                                                   const UDATA_T* inputs,
                                                   unsigned int nbOutputs_,
                                                   unsigned int outputsHeight,
                                                   unsigned int outputsWidth,
                                                   unsigned int nbOutputs,
                                                   unsigned int outputOffset,
                                                   DATA_T* outputs,
                                                   unsigned int poolHeight,
                                                   unsigned int poolWidth,
                                                   Pooling_T pooling,
                                                   ActivationFunction_T func,
                                                   int shift)
{
#if NB_BITS < 0
    if(func != Linear)
        return;
#endif
    const unsigned int batchInputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels * channelsHeight
          * channelsWidth; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z)
          * nbOutputs_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x) {
                const unsigned int sxMax = min(channelsWidth - ox*strideX, (unsigned int) poolWidth);
                const unsigned int syMax = min(channelsHeight - oy*strideY, (unsigned int) poolHeight);

                const unsigned int outputsIdx = ox + (oy + output*outputsHeight)*outputsWidth;

                if (pooling == Max) {
                    UDATA_T poolValue = 0;

                    for (unsigned int sy = 0; sy != syMax; ++sy) {
                        const unsigned int ix = ox*strideX + 0;
                        const unsigned int iy = oy*strideY + sy;
                        const unsigned int inputsIdx = ix + (iy + output*channelsHeight)*channelsWidth;

                        unsigned int sx = 0;
                        for ( ; sx < sxMax; ++sx) {
                            if (inputs[sx + inputsIdx + batchInputOffset] > poolValue)
                                poolValue = inputs[sx + inputsIdx + batchInputOffset];
                        }
                    }

#if NB_BITS < 0
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = usat(poolValue, func, shift);
#else
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = poolValue;
#endif
                }
                else if (pooling == Average) {
                    SUM_T sum = 0;
                    for (unsigned int sy = 0; sy != syMax; ++sy) {
                        const unsigned int ix = ox*strideX + 0;
                        const unsigned int iy = oy*strideY + sy;
                        const unsigned int inputsIdx = ix + (iy + output*channelsHeight)*channelsWidth;

                        for (unsigned int sx = 0; sx < sxMax; ++sx)
                                sum += inputs[sx + inputsIdx + batchInputOffset];
                    }
                    sum /= (poolHeight*poolWidth);
#if NB_BITS < 0
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = usat(sum, func, shift);
#else
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = sum;
#endif
                }
            }
        }
    }
}
__global__ void poolcell_propagate_kernel(unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth,
                                          unsigned int strideY,
                                          unsigned int strideX,
                                          const DATA_T* inputs,
                                          unsigned int nbOutputs_,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int nbOutputs,
                                          unsigned int outputOffset,
                                          DATA_T* outputs,
                                          unsigned int poolHeight,
                                          unsigned int poolWidth,
                                          const char* mapping,
                                          Pooling_T pooling,
                                          ActivationFunction_T func,
                                          int shift)
{
#if NB_BITS < 0
    if(func != Linear)
        return;
#endif
    const unsigned int batchInputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels * channelsHeight
          * channelsWidth; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z)
          * nbOutputs_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x) {
                const unsigned int sxMax = min(channelsWidth - ox*strideX, (unsigned int) poolWidth);
                const unsigned int syMax = min(channelsHeight - oy*strideY, (unsigned int) poolHeight);

                const unsigned int outputsIdx = ox + (oy + output*outputsHeight)*outputsWidth;

                if (pooling == Max) {
                    DATA_T poolValue = DATA_T_MIN;

                    for (unsigned int ch = 0; ch != nbChannels; ++ch) {
                        if (!mapping[ch + output*nbChannels])
                            continue;

                        for (unsigned int sy = 0; sy != syMax; ++sy) {
                            const unsigned int ix = ox*strideX + 0;
                            const unsigned int iy = oy*strideY + sy;
                            const unsigned int inputsIdx = ix + (iy + output*channelsHeight)*channelsWidth;

                            unsigned int sx = 0;

                            for ( ; sx < sxMax; ++sx) {
                                if (inputs[sx + inputsIdx + batchInputOffset] > poolValue)
                                    poolValue = inputs[sx + inputsIdx + batchInputOffset];
                            }
                        }
                    }
#if NB_BITS < 0
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = sat(poolValue, func, shift);
#else
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = poolValue;
#endif

                }
                else if (pooling == Average) {
                    unsigned int nbMapChan = 0;
                    SUM_T sum = 0;
                    for (unsigned int ch = 0; ch != nbChannels; ++ch) {

                        if (!mapping[ch + output*nbChannels])
                            continue;
                        for (unsigned int sy = 0; sy != syMax; ++sy) {
                            const unsigned int ix = ox*strideX + 0;
                            const unsigned int iy = oy*strideY + sy;
                            const unsigned int inputsIdx = ix + (iy + output*channelsHeight)*channelsWidth;

                            for (unsigned int sx = 0; sx < sxMax; ++sx)
                                    sum += inputs[sx + inputsIdx + batchInputOffset];
                        }
                        ++nbMapChan;
                    }
                    sum /= (poolHeight*poolWidth*nbMapChan);
#if NB_BITS < 0
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = sat(sum, func, shift);
#else
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = sum;
#endif
                }
            }
        }
    }
}
__global__ void poolcell_propagate_unitmap_kernel(unsigned int nbChannels,
                                                  unsigned int channelsHeight,
                                                  unsigned int channelsWidth,
                                                  unsigned int strideY,
                                                  unsigned int strideX,
                                                  const DATA_T* inputs,
                                                  unsigned int nbOutputs_,
                                                  unsigned int outputsHeight,
                                                  unsigned int outputsWidth,
                                                  unsigned int nbOutputs,
                                                  unsigned int outputOffset,
                                                  DATA_T* outputs,
                                                  unsigned int poolHeight,
                                                  unsigned int poolWidth,
                                                  Pooling_T pooling,
                                                  ActivationFunction_T func,
                                                  int shift)
{
#if NB_BITS < 0
    if(func != Linear)
        return;
#endif
    const unsigned int batchInputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels * channelsHeight
          * channelsWidth; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z)
          * nbOutputs_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x) {
                const unsigned int sxMax = min(channelsWidth - ox*strideX, (unsigned int) poolWidth);
                const unsigned int syMax = min(channelsHeight - oy*strideY, (unsigned int) poolHeight);

                const unsigned int outputsIdx = ox + (oy + output*outputsHeight)*outputsWidth;

                if (pooling == Max) {
                    DATA_T poolValue = DATA_T_MIN;

                    for (unsigned int sy = 0; sy != syMax; ++sy) {
                        const unsigned int ix = ox*strideX + 0;
                        const unsigned int iy = oy*strideY + sy;
                        const unsigned int inputsIdx = ix + (iy + output*channelsHeight)*channelsWidth;

                        unsigned int sx = 0;

                        for ( ; sx < sxMax; ++sx) {
                            if (inputs[sx + inputsIdx + batchInputOffset] > poolValue)
                                poolValue = inputs[sx + inputsIdx + batchInputOffset];
                        }
                    }

#if NB_BITS < 0
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = sat(poolValue, func, shift);
#else
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = poolValue;
#endif
                }
                else if (pooling == Average) {
                    SUM_T sum = 0;
                    for (unsigned int sy = 0; sy != syMax; ++sy) {
                        const unsigned int ix = ox*strideX + 0;
                        const unsigned int iy = oy*strideY + sy;
                        const unsigned int inputsIdx = ix + (iy + output*channelsHeight)*channelsWidth;

                        for (unsigned int sx = 0; sx < sxMax; ++sx)
                                sum += inputs[sx + inputsIdx + batchInputOffset];
                    }

                    sum /= (poolHeight*poolWidth);
#if NB_BITS < 0
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = sat(sum, func, shift);
#else
                    outputs[batchOutputOffset + outputOffset + outputsIdx] = sum;
#endif
                }
            }
        }
    }
}

__global__ void
fccell_propagate_2d_kernel(unsigned int nbChannels,
                           unsigned int channelsHeight,
                           unsigned int channelsWidth,
                           const DATA_T* inputs,
                           unsigned int nbOutput_,
                           unsigned int nbOutputs,
                           unsigned int outputOffset,
                           DATA_T* outputs,
                           unsigned int nbChannels_,
                           const BDATA_T* bias,
                           const WDATA_T* weights, // FIXME: for GTSRB to work
                           ActivationFunction_T func,
                           int shift)
{
    const unsigned int partialIdx = threadIdx.x;
    const unsigned int partialSize = blockDim.x;
    const unsigned int size = nbChannels * channelsHeight * channelsWidth;
    const unsigned int batchInputOffset
        = (blockDim.y * blockIdx.y + threadIdx.y)
          * size; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset
        = (blockDim.y * blockIdx.y + threadIdx.y)
          * nbOutput_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        const unsigned int offset = output * size;

        extern __shared__ SUM_T partialSum[];

        partialSum[partialIdx] = 0;

        for (unsigned int idx = partialIdx; idx < size; idx += partialSize)
            partialSum[partialIdx] += weights[offset + idx]
                                      * inputs[idx + batchInputOffset];

        __syncthreads();

        // Commutative reduction
        for (int offset = partialSize / 2; offset > 0; offset >>= 1) {
            if (partialIdx < offset)
                partialSum[partialIdx] += partialSum[partialIdx + offset];

            __syncthreads();
        }

        if (partialIdx == 0)
            outputs[batchOutputOffset + outputOffset + output]
                = sat(bias[output] + partialSum[0], func, shift);
    }
}
__global__ void
fccell_upropagate_2d_kernel(unsigned int nbChannels,
                            unsigned int channelsHeight,
                            unsigned int channelsWidth,
                            const UDATA_T* inputs,
                            unsigned int nbOutput_,
                            unsigned int nbOutputs,
                            unsigned int outputOffset,
                            DATA_T* outputs,
                            unsigned int nbChannels_,
                            const BDATA_T* bias,
                            const WDATA_T* weights, // FIXME: for GTSRB to work
                            ActivationFunction_T func,
                            int shift)
{
    const unsigned int partialIdx = threadIdx.x;
    const unsigned int partialSize = blockDim.x;
    const unsigned int size = nbChannels * channelsHeight * channelsWidth;
    const unsigned int batchInputOffset
        = (blockDim.y * blockIdx.y + threadIdx.y)
          * size; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset
        = (blockDim.y * blockIdx.y + threadIdx.y)
          * nbOutput_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        const unsigned int offset = output * size;

        extern __shared__ SUM_T partialSum[];

        partialSum[partialIdx] = 0;

        for (unsigned int idx = partialIdx; idx < size; idx += partialSize)
            partialSum[partialIdx] += weights[offset + idx]
                                      * inputs[idx + batchInputOffset];

        __syncthreads();

        // Commutative reduction
        for (int offset = partialSize / 2; offset > 0; offset >>= 1) {
            if (partialIdx < offset)
                partialSum[partialIdx] += partialSum[partialIdx + offset];

            __syncthreads();
        }

        if (partialIdx == 0)
            outputs[batchOutputOffset + outputOffset + output]
                = usat(bias[output] + partialSum[0], func, shift);
    }
}
__global__ void fccell_propagate_kernel(unsigned int nbChannels,
                                        const DATA_T* inputs,
                                        unsigned int nbOutputs_,
                                        unsigned int nbOutputs,
                                        unsigned int outputOffset,
                                        DATA_T* outputs,
                                        const BDATA_T* bias,
                                        const WDATA_T* weights,
                                        ActivationFunction_T func,
                                        int shift)
{
    const unsigned int partialIdx = threadIdx.x;
    const unsigned int partialSize = blockDim.x;
    const unsigned int batchInputOffset
        = (blockDim.y * blockIdx.y + threadIdx.y)
          * nbChannels; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset
        = (blockDim.y * blockIdx.y + threadIdx.y)
          * nbOutputs_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        const unsigned int offset = output * nbChannels;

        extern __shared__ SUM_T partialSum[];
        partialSum[partialIdx] = 0;

        for (unsigned int ch = partialIdx; ch < nbChannels; ch += partialSize)
            partialSum[partialIdx] += weights[offset + ch]
                                      * inputs[ch + batchInputOffset];

        __syncthreads();

        // Commutative reduction
        for (int offset = partialSize / 2; offset > 0; offset >>= 1) {
            if (partialIdx < offset)
                partialSum[partialIdx] += partialSum[partialIdx + offset];

            __syncthreads();
        }

        if (partialIdx == 0)
            outputs[batchOutputOffset + outputOffset + output]
                = sat(bias[output] + partialSum[0], func, shift);
    }
}

__global__ void fccell_upropagate_kernel(unsigned int nbChannels,
                                        const UDATA_T* inputs,
                                        unsigned int nbOutputs_,
                                        unsigned int nbOutputs,
                                        unsigned int outputOffset,
                                        DATA_T* outputs,
                                        const BDATA_T* bias,
                                        const WDATA_T* weights,
                                        ActivationFunction_T func,
                                        int shift)
{
    const unsigned int partialIdx = threadIdx.x;
    const unsigned int partialSize = blockDim.x;
    const unsigned int batchInputOffset
        = (blockDim.y * blockIdx.y + threadIdx.y)
          * nbChannels; // Determine the input offset brings by batch size
    const unsigned int batchOutputOffset
        = (blockDim.y * blockIdx.y + threadIdx.y)
          * nbOutputs_; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        const unsigned int offset = output * nbChannels;

        extern __shared__ SUM_T partialSum[];
        partialSum[partialIdx] = 0;

        for (unsigned int ch = partialIdx; ch < nbChannels; ch += partialSize)
            partialSum[partialIdx] += weights[offset + ch]
                                      * inputs[ch + batchInputOffset];

        __syncthreads();

        // Commutative reduction
        for (int offset = partialSize / 2; offset > 0; offset >>= 1) {
            if (partialIdx < offset)
                partialSum[partialIdx] += partialSum[partialIdx + offset];

            __syncthreads();
        }

        if (partialIdx == 0)
            outputs[batchOutputOffset + outputOffset + output]
                = usat(bias[output] + partialSum[0], func, shift);
    }
}

__global__ void softmax_propagate_kernel(unsigned int nbOutputs,
                                            unsigned int outputsHeight,
                                            unsigned int outputsWidth,
                                            const DATA_T* inputs,
                                            DATA_T* outputs)
{
#if NB_BITS > 0
     //Copy of the input buffer to the output buffer if type is INT (SoftMax can't be perform on int data type)
    if(nbOutputs > 1) {
        const unsigned int batchOffset
            = (blockIdx.z * blockDim.z + threadIdx.z) * nbOutputs * outputsHeight
              * outputsWidth; // Determine the input offset brings by batch size

        for(unsigned int output = threadIdx.x; output < nbOutputs; output +=  blockDim.x) {
            for(unsigned int oy = blockIdx.y; oy < outputsHeight; oy += gridDim.y) {
                for(unsigned int ox = blockIdx.x; ox < outputsWidth; ox +=  gridDim.x) {
                    unsigned int outputsIdx = ox + (oy + output*outputsHeight)*outputsWidth;
                    outputs[outputsIdx + batchOffset]   = inputs[outputsIdx + batchOffset];
                }
            }
        }
    }
    else {
        const unsigned int batchOffset =(blockIdx.z * blockDim.z + threadIdx.z)*outputsHeight*outputsWidth; //Determine the input offset brings by batch size

        for(unsigned int oy = blockIdx.y; oy < outputsHeight; oy += gridDim.y)
            for(unsigned int ox = blockIdx.x; ox < outputsWidth; ox +=  gridDim.x)
                outputs[ox + oy*outputsWidth + batchOffset] = inputs[ox + oy*outputsWidth + batchOffset];
    }

#else
    if (nbOutputs > 1) {
        const unsigned int batchOffset
            = (blockIdx.z * blockDim.z + threadIdx.z) * nbOutputs * outputsHeight
              * outputsWidth; // Determine the input offset brings by batch size

        DATA_T maxVal = 0;
        double sum = 0;
        DATA_T output_priv = 0;

        for (unsigned int output = threadIdx.x; output < nbOutputs;
             output += blockDim.x) {
            for (unsigned int oy = blockIdx.y; oy < outputsHeight;
                 oy += gridDim.y) {
                for (unsigned int ox = blockIdx.x; ox < outputsWidth;
                     ox += gridDim.x) {
                    extern __shared__ DATA_T local_tmp[];

                    local_tmp[output]
                        = inputs[ox + (oy + output * outputsHeight)
                                       * outputsWidth + batchOffset];
                    __syncthreads();

                    maxVal = local_tmp[0];

                    for(unsigned int output =1; output < nbOutputs; ++output) {
                        if(local_tmp[output] > maxVal) {
                            maxVal = local_tmp[output];
                        }
                    }

                    for (unsigned int output = 0; output < nbOutputs; ++output)
                        sum += exp(local_tmp[output] - maxVal);

                    if (sum > 0.0) {
                        for (unsigned int output = 0; output < nbOutputs; ++output) {
                            output_priv = exp(local_tmp[output] - maxVal)/sum;
                            outputs[output*outputsWidth*outputsHeight + ox + oy*outputsWidth + batchOffset] = output_priv;
                        }

                    }
                    else {
                        for (unsigned int output = 0; output < nbOutputs; ++output)
                            outputs[output*outputsWidth*outputsHeight + ox + oy*outputsWidth + batchOffset] = 0;
                    }
                }
            }
        }
    } else {
        const unsigned int batchOffset = (blockIdx.z * blockDim.z + threadIdx.z)*outputsHeight*outputsWidth; //Determine the input offset brings by batch size

        for (unsigned int oy = blockIdx.y; oy < outputsHeight; oy += gridDim.y)
            for (unsigned int ox = blockIdx.x; ox < outputsWidth;
                 ox += gridDim.x)
                outputs[ox + oy * outputsWidth + batchOffset]
                    = (inputs[ox + oy * outputsWidth + batchOffset]
                       > 0.5);
    }
#endif
}


__global__ void spatial_outputs_max_kernel(unsigned int nbOutputs,
                                            unsigned int outputsHeight,
                                            unsigned int outputsWidth,
                                            const DATA_T* outputs,
                                            DATA_T* outputEstimated)
{
    unsigned int outputMax = 0;
    DATA_T maxVal = 0;

    if (nbOutputs > 1) {

        const unsigned int inputBatchOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbOutputs * outputsHeight
          * outputsWidth; // Determine the input offset brings by batch size

        const unsigned int outputBatchOffset
            = (blockIdx.z * blockDim.z + threadIdx.z) * outputsHeight
              * outputsWidth; // Determine the input offset brings by batch size

        for (unsigned int output = threadIdx.x; output < nbOutputs;
             output += blockDim.x) {
            for (unsigned int oy = blockIdx.y; oy < outputsHeight;
                 oy += gridDim.y) {
                for (unsigned int ox = blockIdx.x; ox < outputsWidth;
                     ox += gridDim.x) {
                    extern __shared__ DATA_T local_tmp[];

                    local_tmp[output]
                        = outputs[ox + (oy + output * outputsHeight)
                                       * outputsWidth + inputBatchOffset];
                    __syncthreads();

                    maxVal = local_tmp[0];
                    for (unsigned int output = 1; output < nbOutputs;
                         ++output) {
                        if (local_tmp[output] > maxVal) {
                            outputMax = output;
                            maxVal = local_tmp[output];
                        }
                    }
                    outputEstimated[ox + oy * outputsWidth + outputBatchOffset]
                        = outputMax;
                }
            }
        }
    } else {
        const unsigned int batchOffset
                = (blockIdx.z * blockDim.z + threadIdx.z)*outputsHeight
                  *outputsWidth; // Determine the input offset brings by batch size

        for (unsigned int oy = blockIdx.y; oy < outputsHeight; oy += gridDim.y)
            for (unsigned int ox = blockIdx.x; ox < outputsWidth;
                 ox += gridDim.x)
                outputEstimated[ox + oy * outputsWidth + batchOffset]
                    = (outputs[ox + oy * outputsWidth + batchOffset]
                       > 0.5);
    }
}

extern "C" void cuda_convcell_propagate(unsigned int nbChannels,
                                        unsigned int channelsHeight,
                                        unsigned int channelsWidth,
                                        unsigned int paddingY,
                                        unsigned int paddingX,
                                        unsigned int strideY,
                                        unsigned int strideX,
                                        unsigned int subSampleY,
                                        unsigned int subSampleX,
                                        const DATA_T* inputs,
                                        unsigned int oySize,
                                        unsigned int oxSize,
                                        unsigned int nbOutputs_,
                                        unsigned int outputsHeight,
                                        unsigned int outputsWidth,
                                        unsigned int nbOutputs,
                                        unsigned int outputOffset,
                                        DATA_T* outputs,
                                        unsigned int kernelHeight,
                                        unsigned int kernelWidth,
                                        const BDATA_T* bias,
                                        const WDATA_T* weights,
                                        ActivationFunction_T func,
                                        int shift,
                                        dim3 threadsPerBlocks,
                                        dim3 blocksPerGrid,
                                        bool isProfiled,
                                        float* exec_time)
{

    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        convcell_propagate_kernel <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
               channelsHeight,
               channelsWidth,
               paddingY,
               paddingX,
               strideY,
               strideX,
               subSampleY,
               subSampleX,
               inputs,
               oySize,
               oxSize,
               nbOutputs_,
               outputsHeight,
               outputsWidth,
               nbOutputs,
               outputOffset,
               outputs,
               kernelHeight,
               kernelWidth,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else
    {
        convcell_propagate_kernel <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
               channelsHeight,
               channelsWidth,
               paddingY,
               paddingX,
               strideY,
               strideX,
               subSampleY,
               subSampleX,
               inputs,
               oySize,
               oxSize,
               nbOutputs_,
               outputsHeight,
               outputsWidth,
               nbOutputs,
               outputOffset,
               outputs,
               kernelHeight,
               kernelWidth,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
    }
}

extern "C" void cuda_convcell_upropagate(unsigned int nbChannels,
                                         unsigned int channelsHeight,
                                         unsigned int channelsWidth,
                                         unsigned int paddingY,
                                         unsigned int paddingX,
                                         unsigned int strideY,
                                         unsigned int strideX,
                                         unsigned int subSampleY,
                                         unsigned int subSampleX,
                                         const UDATA_T* inputs,
                                         unsigned int oySize,
                                         unsigned int oxSize,
                                         unsigned int nbOutputs_,
                                         unsigned int outputsHeight,
                                         unsigned int outputsWidth,
                                         unsigned int nbOutputs,
                                         unsigned int outputOffset,
                                         DATA_T* outputs,
                                         unsigned int kernelHeight,
                                         unsigned int kernelWidth,
                                         const BDATA_T* bias,
                                         const WDATA_T* weights,
                                         ActivationFunction_T func,
                                         int shift,
                                         dim3 threadsPerBlocks,
                                         dim3 blocksPerGrid,
                                         bool isProfiled,
                                         float* exec_time)
{
    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        convcell_upropagate_kernel <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
               channelsHeight,
               channelsWidth,
               paddingY,
               paddingX,
               strideY,
               strideX,
               subSampleY,
               subSampleX,
               inputs,
               oySize,
               oxSize,
               nbOutputs_,
               outputsHeight,
               outputsWidth,
               nbOutputs,
               outputOffset,
               outputs,
               kernelHeight,
               kernelWidth,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else
    {
        convcell_upropagate_kernel <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
               channelsHeight,
               channelsWidth,
               paddingY,
               paddingX,
               strideY,
               strideX,
               subSampleY,
               subSampleX,
               inputs,
               oySize,
               oxSize,
               nbOutputs_,
               outputsHeight,
               outputsWidth,
               nbOutputs,
               outputOffset,
               outputs,
               kernelHeight,
               kernelWidth,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
    }
}
extern "C" void cuda_poolcell_propagate(unsigned int nbChannels,
                                        unsigned int channelsHeight,
                                        unsigned int channelsWidth,
                                        unsigned int strideY,
                                        unsigned int strideX,
                                        const DATA_T* inputs,
                                        unsigned int nbOutputs_,
                                        unsigned int outputsHeight,
                                        unsigned int outputsWidth,
                                        unsigned int nbOutputs,
                                        unsigned int outputOffset,
                                        DATA_T* outputs,
                                        unsigned int poolHeight,
                                        unsigned int poolWidth,
                                        const char* mapping,
                                        Pooling_T pooling,
                                        ActivationFunction_T func,
                                        int shift,
                                        dim3 threadsPerBlocks,
                                        dim3 blocksPerGrid,
                                        bool isProfiled,
                                        float* exec_time,
                                        bool unitMap)
{
    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        if (unitMap == true) {
            poolcell_propagate_unitmap_kernel
                <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
                                                        channelsHeight,
                                                        channelsWidth,
                                                        strideY,
                                                        strideX,
                                                        inputs,
                                                        nbOutputs_,
                                                        outputsHeight,
                                                        outputsWidth,
                                                        nbOutputs,
                                                        outputOffset,
                                                        outputs,
                                                        poolHeight,
                                                        poolWidth,
                                                        pooling,
                                                        func,
                                                        shift);
        } else {
            poolcell_propagate_kernel <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
                   channelsHeight,
                   channelsWidth,
                   strideY,
                   strideX,
                   inputs,
                   nbOutputs_,
                   outputsHeight,
                   outputsWidth,
                   nbOutputs,
                   outputOffset,
                   outputs,
                   poolHeight,
                   poolWidth,
                   mapping,
                   pooling,
                   func,
                   shift);
        }
        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else {
        if (unitMap == true) {

            poolcell_propagate_unitmap_kernel
                <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
                                                        channelsHeight,
                                                        channelsWidth,
                                                        strideY,
                                                        strideX,
                                                        inputs,
                                                        nbOutputs_,
                                                        outputsHeight,
                                                        outputsWidth,
                                                        nbOutputs,
                                                        outputOffset,
                                                        outputs,
                                                        poolHeight,
                                                        poolWidth,
                                                        pooling,
                                                        func,
                                                        shift);
        } else {
            poolcell_propagate_kernel <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
                   channelsHeight,
                   channelsWidth,
                   strideY,
                   strideX,
                   inputs,
                   nbOutputs_,
                   outputsHeight,
                   outputsWidth,
                   nbOutputs,
                   outputOffset,
                   outputs,
                   poolHeight,
                   poolWidth,
                   mapping,
                   pooling,
                   func,
                   shift);
        }
        checkCudaKernelsErrors();
    }
}
extern "C" void cuda_poolcell_upropagate(unsigned int nbChannels,
                                         unsigned int channelsHeight,
                                         unsigned int channelsWidth,
                                         unsigned int strideY,
                                         unsigned int strideX,
                                         const UDATA_T* inputs,
                                         unsigned int nbOutputs_,
                                         unsigned int outputsHeight,
                                         unsigned int outputsWidth,
                                         unsigned int nbOutputs,
                                         unsigned int outputOffset,
                                         DATA_T* outputs,
                                         unsigned int poolHeight,
                                         unsigned int poolWidth,
                                         const char* mapping,
                                         Pooling_T pooling,
                                         ActivationFunction_T func,
                                         int shift,
                                         dim3 threadsPerBlocks,
                                         dim3 blocksPerGrid,
                                         bool isProfiled,
                                         float* exec_time,
                                         bool unitMap)
{
    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        if (unitMap == true) {
            poolcell_upropagate_unitmap_kernel
                <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
                                                        channelsHeight,
                                                        channelsWidth,
                                                        strideY,
                                                        strideX,
                                                        inputs,
                                                        nbOutputs_,
                                                        outputsHeight,
                                                        outputsWidth,
                                                        nbOutputs,
                                                        outputOffset,
                                                        outputs,
                                                        poolHeight,
                                                        poolWidth,
                                                        pooling,
                                                        func,
                                                        shift);
        } else {
            poolcell_upropagate_kernel <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
                   channelsHeight,
                   channelsWidth,
                   strideY,
                   strideX,
                   inputs,
                   nbOutputs_,
                   outputsHeight,
                   outputsWidth,
                   nbOutputs,
                   outputOffset,
                   outputs,
                   poolHeight,
                   poolWidth,
                   mapping,
                   pooling,
                   func,
                   shift);
        }
        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else
    {
        if (unitMap == true) {
            poolcell_upropagate_unitmap_kernel
                <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
                                                        channelsHeight,
                                                        channelsWidth,
                                                        strideY,
                                                        strideX,
                                                        inputs,
                                                        nbOutputs_,
                                                        outputsHeight,
                                                        outputsWidth,
                                                        nbOutputs,
                                                        outputOffset,
                                                        outputs,
                                                        poolHeight,
                                                        poolWidth,
                                                        pooling,
                                                        func,
                                                        shift);
        } else {
            poolcell_upropagate_kernel <<<blocksPerGrid, threadsPerBlocks>>> (nbChannels,
                   channelsHeight,
                   channelsWidth,
                   strideY,
                   strideX,
                   inputs,
                   nbOutputs_,
                   outputsHeight,
                   outputsWidth,
                   nbOutputs,
                   outputOffset,
                   outputs,
                   poolHeight,
                   poolWidth,
                   mapping,
                   pooling,
                   func,
                   shift);
        }
        checkCudaKernelsErrors();
    }
}
extern "C" void cuda_fccell_propagate_2d(unsigned int nbChannels,
                                         unsigned int channelsHeight,
                                         unsigned int channelsWidth,
                                         const DATA_T* inputs,
                                         unsigned int nbOutputs_,
                                         unsigned int nbOutputs,
                                         unsigned int outputOffset,
                                         DATA_T* outputs,
                                         unsigned int nbChannels_,
                                         const BDATA_T* bias,
                                         const WDATA_T* weights,
                                         ActivationFunction_T func,
                                         int shift,
                                         dim3 threadsPerBlocks,
                                         dim3 blocksPerGrid,
                                         bool isProfiled,
                                         float* exec_time)
{
    size_t sharedSize = sizeof(SUM_T) * threadsPerBlocks.x;
    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        fccell_propagate_2d_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (nbChannels,
               channelsHeight,
               channelsWidth,
               inputs,
               nbOutputs_,
               nbOutputs,
               outputOffset,
               outputs,
               nbChannels_,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else
    {
        fccell_propagate_2d_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (nbChannels,
               channelsHeight,
               channelsWidth,
               inputs,
               nbOutputs_,
               nbOutputs,
               outputOffset,
               outputs,
               nbChannels_,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
    }
}

extern "C" void cuda_fccell_upropagate_2d(unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth,
                                          const UDATA_T* inputs,
                                          unsigned int nbOutputs_,
                                          unsigned int nbOutputs,
                                          unsigned int outputOffset,
                                          DATA_T* outputs,
                                          unsigned int nbChannels_,
                                          const BDATA_T* bias,
                                          const WDATA_T* weights,
                                          ActivationFunction_T func,
                                          int shift,
                                          dim3 threadsPerBlocks,
                                          dim3 blocksPerGrid,
                                          bool isProfiled,
                                          float* exec_time)
{
    size_t sharedSize = sizeof(SUM_T) * threadsPerBlocks.x;
    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        fccell_upropagate_2d_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>>
                                    (nbChannels,
                                     channelsHeight,
                                     channelsWidth,
                                     inputs,
                                     nbOutputs_,
                                     nbOutputs,
                                     outputOffset,
                                     outputs,
                                     nbChannels_,
                                     bias,
                                     weights,
                                     func,
                                     shift);
        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else
    {
        fccell_upropagate_2d_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (nbChannels,
                                                             channelsHeight,
                                                             channelsWidth,
                                                             inputs,
                                                             nbOutputs_,
                                                             nbOutputs,
                                                             outputOffset,
                                                             outputs,
                                                             nbChannels_,
                                                             bias,
                                                             weights,
                                                             func,
                                                             shift);
        checkCudaKernelsErrors();
    }
}

extern "C" void cuda_fccell_propagate(unsigned int nbChannels,
                                      const DATA_T* inputs,
                                      unsigned int nbOutputs_,
                                      unsigned int nbOutputs,
                                      unsigned int outputOffset,
                                      DATA_T* outputs,
                                      const BDATA_T* bias,
                                      const WDATA_T* weights,
                                      ActivationFunction_T func,
                                      int shift,
                                      dim3 threadsPerBlocks,
                                      dim3 blocksPerGrid,
                                      bool isProfiled,
                                      float* exec_time)
{
    size_t sharedSize = sizeof(SUM_T) * threadsPerBlocks.x;
    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        fccell_propagate_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (nbChannels,
               inputs,
               nbOutputs_,
               nbOutputs,
               outputOffset,
               outputs,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else
    {
        fccell_propagate_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (nbChannels,
               inputs,
               nbOutputs_,
               nbOutputs,
               outputOffset,
               outputs,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
    }
}

extern "C" void cuda_fccell_upropagate(unsigned int nbChannels,
                                       const UDATA_T* inputs,
                                       unsigned int nbOutputs_,
                                       unsigned int nbOutputs,
                                       unsigned int outputOffset,
                                       DATA_T* outputs,
                                       const BDATA_T* bias,
                                       const WDATA_T* weights,
                                       ActivationFunction_T func,
                                       int shift,
                                       dim3 threadsPerBlocks,
                                       dim3 blocksPerGrid,
                                       bool isProfiled,
                                       float* exec_time)
{
    size_t sharedSize = sizeof(SUM_T) * threadsPerBlocks.x;
    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        fccell_upropagate_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (nbChannels,
               inputs,
               nbOutputs_,
               nbOutputs,
               outputOffset,
               outputs,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else
    {
        fccell_upropagate_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (nbChannels,
               inputs,
               nbOutputs_,
               nbOutputs,
               outputOffset,
               outputs,
               bias,
               weights,
               func,
               shift);
        checkCudaKernelsErrors();
    }
}

extern "C" void cuda_softmaxcell_propagate(unsigned int nbOutputs,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int batchSize,
                                          const DATA_T* inputs,
                                          DATA_T* outputs,
                                          dim3 threadsPerBlocks,
                                          dim3 blocksPerGrid,
                                          bool isProfiled,
                                          float* exec_time)
{
    size_t sharedSize = sizeof(DATA_T) * nbOutputs;
    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        softmax_propagate_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (nbOutputs, outputsHeight, outputsWidth, inputs, outputs);
        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else {
        softmax_propagate_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>>
        (nbOutputs,
         outputsHeight,
         outputsWidth,
         inputs,
         outputs);

        checkCudaKernelsErrors();
    }
}

extern "C" void cuda_spatial_outputs_max(unsigned int nbOutputs,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int batchSize,
                                          DATA_T* inputs,
                                          DATA_T* outputs,
                                          dim3 threadsPerBlocks,
                                          dim3 blocksPerGrid,
                                          bool isProfiled,
                                          float* exec_time)
{
    size_t sharedSize = sizeof(DATA_T) * nbOutputs;
    if(isProfiled) {
        cudaEvent_t start, stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        spatial_outputs_max_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>>
            (nbOutputs,
             outputsHeight,
             outputsWidth,
             inputs,
             outputs);

        checkCudaKernelsErrors();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        *exec_time = elapsedTimeInMs;
    }
    else {
        spatial_outputs_max_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>>
            (nbOutputs,
             outputsHeight,
             outputsWidth,
             inputs,
             outputs);

        checkCudaKernelsErrors();
    }
}
