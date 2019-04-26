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
    unsigned int outputMax = 0;
    float maxVal = 0.0f;

	const int index = threadIdx.x + blockIdx.x*blockDim.x;
	const int probabilityMapSize = targetWidth*targetHeight*nbClass;
    const int batchInputOffset = (nbClass > 1) 
                                    ? probabilityMapSize*blockIdx.z
                                    : targetWidth*targetHeight*blockIdx.z;

    const int batchOutputOffset = targetWidth*targetHeight*topN*blockIdx.z;

    if (index < targetWidth*targetHeight) {
        if(nbClass > 1)
        {
            //__syncthreads();
            if(topN > 1)
            {
                extern __shared__ float local_tmp[];
                for(unsigned int cls = 0; cls < nbClass; ++cls)
                    local_tmp[threadIdx.x*nbClass + cls]
                        = input[index + cls*targetWidth*targetHeight + batchInputOffset];

                float tmpValue = 0.0f;
                int tmpIdx = 0;

                int* idxData = (int*)&local_tmp[groupSize*nbClass]; // nF floats
                for(unsigned int cls = 0; cls < nbClass; ++cls)
                    idxData[threadIdx.x*nbClass + cls] = cls;

                //Sorting in a descending order
                for (int i = 0; i < nbClass; ++i) {
                    for (int j = 0; j < nbClass; ++j) {
                        if(local_tmp[threadIdx.x*nbClass + j] 
                                < local_tmp[threadIdx.x*nbClass + i])
                        {
                            tmpValue 
                                = local_tmp[threadIdx.x*nbClass + i];
                            local_tmp[threadIdx.x*nbClass + i] 
                                = local_tmp[threadIdx.x*nbClass + j];
                            local_tmp[threadIdx.x*nbClass + j] 
                                = tmpValue;

                            tmpIdx = idxData[threadIdx.x*nbClass + i];
                            idxData[threadIdx.x*nbClass + i] 
                                = idxData[threadIdx.x*nbClass + j];
                            idxData[threadIdx.x*nbClass + j] = tmpIdx;
                        }
                    }
                }

                //Write to output
                for (unsigned int cls = 0; cls < topN; ++cls) 
                {
                    estimatedLabelsValue[index + cls*targetWidth*targetHeight + batchOutputOffset] 
                            = local_tmp[threadIdx.x*nbClass + cls];
                    estimatedLabels[index + cls*targetWidth*targetHeight + batchOutputOffset] 
                            = idxData[threadIdx.x*nbClass + cls];

                }

            }
            else
            {
                maxVal = input[index + batchInputOffset];

                for (unsigned int cls = 1; cls < nbClass; ++cls) {
                    float tmp = input[index + cls*targetWidth*targetHeight + batchInputOffset];
                    if (tmp > maxVal) {
                        outputMax = cls;
                        maxVal = tmp;
                    }
                    
                }
                estimatedLabels[index + batchOutputOffset] = outputMax;
                estimatedLabelsValue[index + batchOutputOffset] = maxVal;

            }
        }
        else if(nbClass == 1)
        {
            if(input[index + batchInputOffset] > threshold)
            {
                outputMax = 1;
                maxVal = input[index + batchInputOffset];
            }
            estimatedLabels[index + batchOutputOffset] = outputMax;
            estimatedLabelsValue[index + batchOutputOffset] = maxVal;
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


