/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#include "Target/Target_CUDA_kernels.hpp"

__global__
void cudaGetEstimatedTarget_kernel( const unsigned int topN,
                                    const unsigned int nbClass,
                                    const unsigned int targetHeight,
                                    const unsigned int targetWidth,
                                    const unsigned int batchSize,
                                    const unsigned int groupSize,
                                    float threshold,
                                    const float* input,
                                    float* estimatedLabelsValue,
                                    int* estimatedLabels)
{
    const int batchInputOffset = targetWidth*targetHeight*nbClass*blockIdx.z;
    const int batchOutputOffset = targetWidth*targetHeight*topN*blockIdx.z;

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < targetWidth * targetHeight; i += stride) {
        if(nbClass > 1) {
            //__syncthreads();
            if(topN > 1) {
                extern __shared__ float local_tmp[];
                for(unsigned int cls = 0; cls < nbClass; ++cls)
                    local_tmp[threadIdx.x*nbClass + cls]
                        = input[i + cls*targetWidth*targetHeight + batchInputOffset];

                float tmpValue = 0.0f;
                int tmpIdx = 0;

                int* idxData = (int*)&local_tmp[groupSize*nbClass]; // nF floats
                for(unsigned int cls = 0; cls < nbClass; ++cls)
                    idxData[threadIdx.x*nbClass + cls] = cls;

                //Sorting in a descending order
                for (int x = 0; x < nbClass; ++x) {
                    for (int y = 0; y < nbClass; ++y) {
                        if(local_tmp[threadIdx.x*nbClass + y] 
                                < local_tmp[threadIdx.x*nbClass + x])
                        {
                            tmpValue 
                                = local_tmp[threadIdx.x*nbClass + x];
                            local_tmp[threadIdx.x*nbClass + x] 
                                = local_tmp[threadIdx.x*nbClass + y];
                            local_tmp[threadIdx.x*nbClass + y] 
                                = tmpValue;

                            tmpIdx = idxData[threadIdx.x*nbClass + x];
                            idxData[threadIdx.x*nbClass + x] 
                                = idxData[threadIdx.x*nbClass + y];
                            idxData[threadIdx.x*nbClass + y] = tmpIdx;
                        }
                    }
                }

                //Write to output
                for (unsigned int cls = 0; cls < topN; ++cls) 
                {
                    estimatedLabelsValue[i + cls*targetWidth*targetHeight + batchOutputOffset] 
                            = local_tmp[threadIdx.x*nbClass + cls];
                    estimatedLabels[i + cls*targetWidth*targetHeight + batchOutputOffset] 
                            = idxData[threadIdx.x*nbClass + cls];

                }

            }
            else {
                unsigned int outputMax = 0;
                float maxVal = input[i + batchInputOffset];

                for (unsigned int cls = 1; cls < nbClass; ++cls) {
                    const float tmp = input[i + cls*targetWidth*targetHeight
                                            + batchInputOffset];

                    if (tmp > maxVal) {
                        outputMax = cls;
                        maxVal = tmp;
                    }
                }

                estimatedLabels[i + batchOutputOffset] = outputMax;
                estimatedLabelsValue[i + batchOutputOffset] = maxVal;

            }
        }
        else if (nbClass == 1) {
            const int estimatedLabel
                = (input[i + batchInputOffset] > threshold);

            estimatedLabels[i + batchOutputOffset] = estimatedLabel;
            estimatedLabelsValue[i + batchOutputOffset]
                = (estimatedLabel == 1)
                    ? input[i + batchInputOffset]
                    : (1.0 - input[i + batchInputOffset]);
        }
    }
}

__global__
void cudaGetEstimatedLabel_kernel(const float* value,
                                  unsigned int outputWidth,
                                  unsigned int outputHeight,
                                  unsigned int nbOutputs,
                                  unsigned int batchPos,
                                  unsigned int x0,
                                  unsigned int x1,
                                  unsigned int y0,
                                  unsigned int y1,
                                  float* bbLabels,
                                  const int* mask,
                                  int maskedLabel,
                                  const float* maskValue)
{
    const unsigned int batchOffset
        = outputWidth * outputHeight * nbOutputs * batchPos;
    const unsigned int dimZ = (nbOutputs > 1) ? nbOutputs : 2;

    for (unsigned int z = blockIdx.x; z < dimZ; z += gridDim.x) {
        __shared__ unsigned int count;

        if (threadIdx.x == 0 && threadIdx.y == 0)
            count = 0;

        __syncthreads();

        for (unsigned int y = y0 + threadIdx.y; y < y1; y += blockDim.y) {
            for (unsigned int x = x0 + threadIdx.x; x < x1; x += blockDim.x) {
                const unsigned int idx = x + outputWidth
                    * (y + outputHeight * z * (nbOutputs > 1)) + batchOffset;
                const unsigned int maskIdx = x + outputWidth * y;

                if (mask == NULL || mask[maskIdx] == maskedLabel) {
                    float val = (nbOutputs > 1 || z > 0)
                        ? value[idx]
                        // nbOutputs == 1 && z == 0
                        : 1.0f - value[idx];

                    if (maskValue != NULL)
                        val *= maskValue[maskIdx];

                    atomicAdd(bbLabels + z, val);
                    atomicAdd(&count, 1);
                }
            }
        }

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            bbLabels[z] /= count;
        }
    }
}

void N2D2::cudaGetEstimatedTarget(const unsigned int topN,
                                  const unsigned int nbClass,
                                  const unsigned int targetHeight,
                                  const unsigned int targetWidth,
                                  const unsigned int batchSize,
                                  float threshold,
                                  const float* input,
                                  float* estimatedLabelsValue,
                                  int* estimatedLabels)
{
    const unsigned int groupSize = min(32, targetHeight*targetWidth) ;
    const unsigned int blockSize = ceil(targetHeight*targetWidth / groupSize) ;

    const dim3 threadsPerBlocks = {groupSize, 1, 1};
    const dim3 blocksPerGrid = {blockSize , 1, batchSize};

    size_t sharedSizeF = topN > 1 ? sizeof(float) * nbClass * groupSize : 0;
    size_t sharedSizeI = topN > 1 ? sizeof(int) * nbClass * groupSize : 0;
    size_t totalSharedSize = sharedSizeI + sharedSizeF;


    cudaGetEstimatedTarget_kernel<<<blocksPerGrid, threadsPerBlocks, totalSharedSize>>>
        (topN,
           nbClass,
           targetHeight,
           targetWidth,
           batchSize,
           groupSize,
           threshold,
           input,
           estimatedLabelsValue,
           estimatedLabels);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaGetEstimatedLabel(const cudaDeviceProp& deviceProp,
                                 const float* value,
                                 unsigned int outputWidth,
                                 unsigned int outputHeight,
                                 unsigned int nbOutputs,
                                 unsigned int batchPos,
                                 unsigned int x0,
                                 unsigned int x1,
                                 unsigned int y0,
                                 unsigned int y1,
                                 float* bbLabels,
                                 const int* mask,
                                 int maskedLabel,
                                 const float* maskValue)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int dimZ = (nbOutputs > 1) ? nbOutputs : 2;
    const unsigned int sizeX = (x1 - x0);
    const unsigned int sizeY = (y1 - y0);

    const unsigned int groupSize = (sizeX * sizeY < maxSize)
                                       ? sizeX * sizeY
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)sizeX);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {dimZ, 1, 1};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaGetEstimatedLabel_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (value,
           outputWidth,
           outputHeight,
           nbOutputs,
           batchPos,
           x0,
           x1,
           y0,
           y1,
           bbLabels,
           mask,
           maskedLabel,
           maskValue);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}



