/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Cell/ROIPoolingCell_Frame_CUDA_Kernels.hpp"
//Utils
__device__ __inline__ float fclampf(float x, float min, float max)
{
    return (x < min) ? min : (x > max) ? max : x;
}

// Forward
__global__
void cudaSROIPoolingForwardBilinearTF_kernel(const float alpha,
                                          float* proposals,
                                          unsigned int nbProposals,
                                          unsigned int inputSizeY,
                                          unsigned int inputSizeX,
                                          float* inputs,
                                          unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth,
                                          unsigned int batchSize,
                                          const float beta,
                                          float* outputs,
                                          unsigned int nbOutputs,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int outputOffset,
                                          bool bilinearTF,
                                          bool ignorePad,
                                          float xOffset,
                                          float yOffset,
                                          float xRatio,
                                          float yRatio)
{
    const unsigned int batchProposalsOffset = blockIdx.z * 4;

    const unsigned int batchInputOffset = (blockIdx.z / nbProposals)
                                * nbChannels * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    float x = (proposals[0 + batchProposalsOffset]) / xRatio - xOffset;
    float y = (proposals[1 + batchProposalsOffset]) / yRatio - yOffset;
    float w = (proposals[2 + batchProposalsOffset]) / xRatio;
    float h = (proposals[3 + batchProposalsOffset]) / yRatio;

    // Crop ROI to image boundaries
    if (x < 0) {
        w+= x;
        x = 0;
    }
    if (y < 0) {
        h+= y;
        y = 0;
    }
    if (x + w > (int)channelsWidth)
        w = channelsWidth - x ;
    if (y + h > (int)channelsHeight)
        h = channelsHeight - y ;

    float xPoolRatio, yPoolRatio;

    if (bilinearTF) {
        xPoolRatio = w / (outputsWidth - 1);
        yPoolRatio = h / (outputsHeight - 1);
    }
    else {
        xPoolRatio = w / outputsWidth;
        yPoolRatio = h / outputsHeight;
    }

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                float sx, sy;

                if (bilinearTF) {
                    sx = fminf(x + ox * xPoolRatio, channelsWidth - 1);
                    sy = fminf(y + oy * yPoolRatio, channelsHeight - 1);
                }
                else {
                    // -0.5 + (ox + 0.5) and not ox because the
                    // interpolation is done relative to the CENTER
                    // of the pixels
                    sx = x + fclampf( -0.5 + (ox + 0.5) * xPoolRatio, 0, w - 1);
                    sy = y + fclampf( -0.5 + (oy + 0.5) * yPoolRatio, 0, h - 1);
                }

                const unsigned int sx0 = (int)(sx);
                const unsigned int sy0 = (int)(sy);

                const float dx = sx - sx0;
                const float dy = sy - sy0;

                const unsigned int idxI00 = sx0 + sy0*channelsWidth
                                            + channel*channelsHeight*channelsWidth
                                            + batchInputOffset;

                const unsigned int idxI10 = (sx0 + 1) + sy0*channelsWidth
                                            + channel*channelsHeight*channelsWidth
                                            + batchInputOffset;

                const unsigned int idxI01 = sx0 + (sy0 + 1)*channelsWidth
                                            + channel*channelsHeight*channelsWidth
                                            + batchInputOffset;

                const unsigned int idxI11 = (sx0 + 1) + (sy0 + 1)*channelsWidth
                                            + channel*channelsHeight*channelsWidth
                                            + batchInputOffset;
                const bool invalid = ignorePad ? (((sx0 + 1 < channelsWidth )  && (sy0 + 1 < channelsHeight ))  ? false : true) : false;
                                            
/**INITIAL
                const float i00 = inputs[idxI00];

                const float i10 = (sx0 + 1 < channelsWidth ) ?
                                     inputs[idxI10] : 0.0;

                const float i01 = (sy0 + 1 < channelsHeight ) ?
                                     inputs[idxI01]: 0.0;

                const float i11 = (sx0 + 1 < channelsWidth 
                                     && sy0 + 1 < channelsHeight )
                                     ? inputs[idxI11] : 0.0;
**/
                const float i00 = (!invalid) ? inputs[idxI00] : 0.0;

                const float i10 = (sx0 + 1 < channelsWidth ) && (!invalid) ?
                                     inputs[idxI10] : 0.0;

                const float i01 = (sy0 + 1 < channelsHeight ) && (!invalid)  ?
                                     inputs[idxI01]: 0.0;

                const float i11 = (sx0 + 1 < channelsWidth 
                                     && sy0 + 1 < channelsHeight ) && (!invalid) 
                                     ? inputs[idxI11] : 0.0;


                const float value
                    = i00 * (1 - dx) * (1 - dy)
                    + i10 * dx * (1 - dy)
                    + i01 * (1 - dx) * dy
                    + i11 * (dx * dy);


                const unsigned int outputsIdx
                    = ox + (oy + (channel + outputOffset) * outputsHeight)
                        * outputsWidth + batchOutputOffset;

                outputs[outputsIdx]
                    = alpha * value + beta * outputs[outputsIdx];
            }
        }
    }
}
// Forward
__global__
void cudaSROIPoolingForwardAverage_kernel(const float alpha,
                                          float* proposals,
                                          unsigned int nbProposals,
                                          unsigned int inputSizeY,
                                          unsigned int inputSizeX,
                                          float* inputs,
                                          unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth,
                                          unsigned int batchSize,
                                          const float beta,
                                          float* outputs,
                                          unsigned int nbOutputs,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int outputOffset)
{
    const unsigned int batchProposalsOffset = blockIdx.z * 4;
    const unsigned int batchInputOffset = (blockIdx.z / nbProposals)
                                * nbChannels * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    const float xRatio = ceil(inputSizeX / (float)channelsWidth);
    const float yRatio = ceil(inputSizeY / (float)channelsHeight);

    float x = proposals[0 + batchProposalsOffset] / xRatio;
    float y = proposals[1 + batchProposalsOffset] / yRatio;
    float w = proposals[2 + batchProposalsOffset] / xRatio;
    float h = proposals[3 + batchProposalsOffset] / yRatio;

    // Crop ROI to image boundaries
    if (x < 0) {
        w+= x;
        x = 0;
    }
    if (y < 0) {
        h+= y;
        y = 0;
    }
    if (x + w > (int)channelsWidth)
        w = channelsWidth - x;
    if (y + h > (int)channelsHeight)
        h = channelsHeight - y;

    const float poolWidth = w / outputsWidth;
    const float poolHeight = h / outputsHeight;

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                const unsigned int sxMin = (unsigned int)(x
                                        + ox * poolWidth);
                const unsigned int sxMax = (unsigned int)(x
                                        + (ox + 1) * poolWidth);
                const unsigned int syMin = (unsigned int)(y
                                        + oy * poolHeight);
                const unsigned int syMax = (unsigned int)(y
                                        + (oy + 1) * poolHeight);

                // For each channel, compute the pool value
                float poolValue = 0.0;
                unsigned int poolCount = 0;

                for (unsigned int sy = syMin; sy < syMax; ++sy) {
                    for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                        const unsigned int inputsIdx
                            = sx
                                + (sy + channel * channelsHeight)
                                    * channelsWidth;

                        poolValue += inputs[inputsIdx + batchInputOffset];
                    }
                }

                poolCount += (sxMax - sxMin)*(syMax - syMin);

                const unsigned int outputsIdx
                    = ox + (oy + (channel + outputOffset) * outputsHeight)
                        * outputsWidth + batchOutputOffset;
                outputs[outputsIdx]
                    = alpha * ((poolCount > 0) ?
                                  (poolValue / poolCount) : 0.0)
                      + beta * outputs[outputsIdx];
            }
        }
    }
}

__global__
void cudaSROIPoolingForwardMax_kernel(const float alpha,
                                      float* proposals,
                                      unsigned int nbProposals,
                                      unsigned int inputSizeY,
                                      unsigned int inputSizeX,
                                      float* inputs,
                                      unsigned int nbChannels,
                                      unsigned int channelsHeight,
                                      unsigned int channelsWidth,
                                      unsigned int batchSize,
                                      const float beta,
                                      float* outputs,
                                      unsigned int nbOutputs,
                                      unsigned int outputsHeight,
                                      unsigned int outputsWidth,
                                      unsigned int outputOffset,
                                      N2D2::PoolCell_Frame_Kernels::ArgMax*
                                        argMax)
{
    const unsigned int batchProposalsOffset = blockIdx.z * 4;
    const unsigned int batchInputOffset = (blockIdx.z / nbProposals)
                                * nbChannels * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    const float xRatio = ceil(inputSizeX / (float)channelsWidth);
    const float yRatio = ceil(inputSizeY / (float)channelsHeight);

    float x = proposals[0 + batchProposalsOffset] / xRatio;
    float y = proposals[1 + batchProposalsOffset] / yRatio;
    float w = proposals[2 + batchProposalsOffset] / xRatio;
    float h = proposals[3 + batchProposalsOffset] / yRatio;

    // Crop ROI to image boundaries
    if (x < 0) {
        w+= x;
        x = 0;
    }
    if (y < 0) {
        h+= y;
        y = 0;
    }
    if (x + w > (int)channelsWidth)
        w = channelsWidth - x;
    if (y + h > (int)channelsHeight)
        h = channelsHeight - y;

    const float poolWidth = w / outputsWidth;
    const float poolHeight = h / outputsHeight;

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                const unsigned int sxMin = (unsigned int)(x
                                        + ox * poolWidth);
                const unsigned int sxMax = (unsigned int)(x
                                        + (ox + 1) * poolWidth);
                const unsigned int syMin = (unsigned int)(y
                                        + oy * poolHeight);
                const unsigned int syMax = (unsigned int)(y
                                        + (oy + 1) * poolHeight);

                // For each channel, compute the pool value
                float poolValue = 0.0;

                const unsigned int argMaxIdx
                    = ox + (oy + channel * outputsHeight)
                        * outputsWidth + batchOutputOffset;
                const unsigned int outputsIdx = argMaxIdx
                    + outputOffset * (outputsHeight * outputsWidth);

                unsigned int ixMax = 0;
                unsigned int iyMax = 0;
                bool valid = false;

                for (unsigned int sy = syMin; sy < syMax; ++sy) {
                    for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                        const unsigned int inputsIdx
                            = sx
                                + (sy + channel * channelsHeight)
                                    * channelsWidth;

                        const float value = inputs[inputsIdx
                                                + batchInputOffset];

                        if (!valid || value > poolValue) {
                            poolValue = value;
                            valid = true;

                            ixMax = sx;
                            iyMax = sy;
                        }
                    }
                }

                argMax[argMaxIdx].ix = ixMax;
                argMax[argMaxIdx].iy = iyMax;
                argMax[argMaxIdx].channel = channel;
                argMax[argMaxIdx].valid = valid;

                outputs[outputsIdx]
                    = alpha * poolValue
                      + beta * outputs[outputsIdx];
            }
        }
    }
}

// Backward
__global__
void cudaSROIPoolingBackwardAverage_kernel(const float alpha,
                                          float* proposals,
                                          unsigned int nbProposals,
                                          unsigned int inputSizeY,
                                          unsigned int inputSizeX,
                                          float* diffInputs,
                                          unsigned int nbOutputs,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int batchSize,
                                          unsigned int outputOffset,
                                          const float beta,
                                          float* diffOutputs,
                                          unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth)
{
    //TODO
}

__global__
void cudaSROIPoolingBackwardMax_kernel(const float alpha,
                                      float* proposals,
                                      unsigned int nbProposals,
                                      unsigned int inputSizeY,
                                      unsigned int inputSizeX,
                                      float* diffInputs,
                                      unsigned int nbOutputs,
                                      unsigned int outputsHeight,
                                      unsigned int outputsWidth,
                                      unsigned int batchSize,
                                      unsigned int outputOffset,
                                      const float /*beta*/,
                                      float* diffOutputs,
                                      unsigned int nbChannels,
                                      unsigned int channelsHeight,
                                      unsigned int channelsWidth,
                                      N2D2::PoolCell_Frame_Kernels::ArgMax*
                                        argMax)
{
    const unsigned int batchProposalsOffset = blockIdx.z * 4;
    const unsigned int batchInputOffset = (blockIdx.z / nbProposals)
                                * nbChannels * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;
    //const float betaBatch = (blockIdx.z % nbProposals == 0)
    //                            ? beta : 1.0;

    const float xRatio = ceil(inputSizeX / (float)channelsWidth);
    const float yRatio = ceil(inputSizeY / (float)channelsHeight);

    float x = proposals[0 + batchProposalsOffset] / xRatio;
    float y = proposals[1 + batchProposalsOffset] / yRatio;
    float w = proposals[2 + batchProposalsOffset] / xRatio;
    float h = proposals[3 + batchProposalsOffset] / yRatio;

    // Crop ROI to image boundaries
    if (x < 0) {
        w+= x;
        x = 0;
    }
    if (y < 0) {
        h+= y;
        y = 0;
    }
    if (x + w > (int)channelsWidth)
        w = channelsWidth - x;
    if (y + h > (int)channelsHeight)
        h = channelsHeight - y;

    const float poolWidth = w / outputsWidth;
    const float poolHeight = h / outputsHeight;

    const unsigned int ixMin = (unsigned int)(x);
    const unsigned int ixMax = (unsigned int)(x + w);
    const unsigned int iyMin = (unsigned int)(y);
    const unsigned int iyMax = (unsigned int)(y + h);

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x)
    {
        for (unsigned int iy = threadIdx.y; iy < channelsHeight;
            iy += blockDim.y)
        {
            for (unsigned int ix = threadIdx.x; ix < channelsWidth;
                ix += blockDim.x)
            {
                if (ix >= ixMin && ix < ixMax
                    && iy >= iyMin && iy < iyMax)
                {
                    const unsigned int ox
                        = (unsigned int)((ix - ixMin + 0.5) / poolWidth);
                    const unsigned int oy
                        = (unsigned int)((iy - iyMin + 0.5) / poolHeight);

                    const unsigned int argMaxIdx
                        = ox + (oy + channel * outputsHeight)
                            * outputsWidth + batchOutputOffset;
                    const unsigned int outputsIdx = argMaxIdx
                        + outputOffset * (outputsHeight * outputsWidth);
                    const N2D2::PoolCell_Frame_Kernels::ArgMax inputMax
                        = argMax[argMaxIdx];

                    if (ix == inputMax.ix
                        && iy == inputMax.iy
                        && inputMax.valid)
                    {
                        const unsigned int inputsIdx
                            = ix + (iy + channel * channelsHeight)
                                * channelsWidth + batchInputOffset;

                        atomicAdd(diffOutputs + inputsIdx,
                                  alpha * diffInputs[outputsIdx]);
                    }
                }
/*
                diffOutputs[inputsIdx]
                    = alpha * poolGradient
                      + betaBatch * diffOutputs[inputsIdx];
*/
            }
        }
    }
}

static unsigned int nextDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        ++v;
    return v;
}

void N2D2::cudaSROIPoolingForwardBilinear(const float alpha,
                                   float* proposals,
                                   unsigned int nbProposals,
                                   unsigned int inputSizeY,
                                   unsigned int inputSizeX,
                                   float* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   const float beta,
                                   float* outputs,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth,
                                   unsigned int outputOffset,
                                   bool bilinearTF,
                                   bool ignorePad,
                                   float xOffset,
                                   float yOffset,
                                   float xRatio,
                                   float yRatio)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, outputsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSROIPoolingForwardBilinearTF_kernel <<<blocksPerGrid, threadsPerBlocks>>
        >(alpha,
          proposals,
          nbProposals,
          inputSizeY,
          inputSizeX,
          inputs,
          nbChannels,
          channelsHeight,
          channelsWidth,
          batchSize,
          beta,
          outputs,
          nbOutputs,
          outputsHeight,
          outputsWidth,
          outputOffset,
          bilinearTF,
          ignorePad,
          xOffset,
          yOffset,
          xRatio,
          yRatio);
}

void N2D2::cudaSROIPoolingForwardAverage(const float alpha,
                                         float* proposals,
                                         unsigned int nbProposals,
                                         unsigned int inputSizeY,
                                         unsigned int inputSizeX,
                                         float* inputs,
                                         unsigned int nbChannels,
                                         unsigned int channelsHeight,
                                         unsigned int channelsWidth,
                                         unsigned int batchSize,
                                         const float beta,
                                         float* outputs,
                                         unsigned int nbOutputs,
                                         unsigned int outputsHeight,
                                         unsigned int outputsWidth,
                                         unsigned int outputOffset)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, outputsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSROIPoolingForwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           proposals,
           nbProposals,
           inputSizeY,
           inputSizeX,
           inputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           beta,
           outputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           outputOffset);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSROIPoolingForwardMax(const float alpha,
                                     float* proposals,
                                     unsigned int nbProposals,
                                     unsigned int inputSizeY,
                                     unsigned int inputSizeX,
                                     float* inputs,
                                     unsigned int nbChannels,
                                     unsigned int channelsHeight,
                                     unsigned int channelsWidth,
                                     unsigned int batchSize,
                                     const float beta,
                                     float* outputs,
                                     unsigned int nbOutputs,
                                     unsigned int outputsHeight,
                                     unsigned int outputsWidth,
                                     unsigned int outputOffset,
                                     N2D2::PoolCell_Frame_Kernels::ArgMax*
                                        argMax)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, outputsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSROIPoolingForwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           proposals,
           nbProposals,
           inputSizeY,
           inputSizeX,
           inputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           beta,
           outputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           outputOffset,
           argMax);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSROIPoolingBackwardAverage(const float alpha,
                                          float* proposals,
                                          unsigned int nbProposals,
                                          unsigned int inputSizeY,
                                          unsigned int inputSizeX,
                                          float* diffInputs,
                                          unsigned int nbOutputs,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int batchSize,
                                          unsigned int outputOffset,
                                          const float beta,
                                          float* diffOutputs,
                                          unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, channelsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSROIPoolingBackwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           proposals,
           nbProposals,
           inputSizeY,
           inputSizeX,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           outputOffset,
           beta,
           diffOutputs,
           nbChannels,
           channelsHeight,
           channelsWidth);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSROIPoolingBackwardMax(const float alpha,
                                      float* proposals,
                                      unsigned int nbProposals,
                                      unsigned int inputSizeY,
                                      unsigned int inputSizeX,
                                      float* diffInputs,
                                      unsigned int nbOutputs,
                                      unsigned int outputsHeight,
                                      unsigned int outputsWidth,
                                      unsigned int batchSize,
                                      unsigned int outputOffset,
                                      const float beta,
                                      float* diffOutputs,
                                      unsigned int nbChannels,
                                      unsigned int channelsHeight,
                                      unsigned int channelsWidth,
                                      N2D2::PoolCell_Frame_Kernels::ArgMax*
                                        argMax)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, channelsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSROIPoolingBackwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           proposals,
           nbProposals,
           inputSizeY,
           inputSizeX,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           outputOffset,
           beta,
           diffOutputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           argMax);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
