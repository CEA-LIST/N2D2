/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes Thiele (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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


#include "CMonitor_CUDA_kernels.hpp"


__global__ void cudaUpdateMetrics_kernel(float * inputs,
                                        int * activity,
                                        long long unsigned int * firingRate, 
                                        long long unsigned int * totalFiringRate,
                                        long long int * outputsActivity, 
                                        long long int * totalOutputsActivity, 
                                        unsigned int inputsDimX,
                                        unsigned int inputsDimY,
                                        unsigned int inputsDimZ)
{
    const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;
    const unsigned int batchInputOffset = blockIdx.z * inputSize;

    for (unsigned int channel = blockIdx.x; channel < inputsDimZ; channel += gridDim.x) {
        for (unsigned int y = threadIdx.y; y < inputsDimY; y += blockDim.y) {
            for (unsigned int x = threadIdx.x; x < inputsDimX; x += blockDim.x) {

                    const unsigned int inputsIdx =
                        x + y*inputsDimX + channel*inputsDimX*inputsDimY;
                    
                    int value = round(inputs[inputsIdx + batchInputOffset]);
                    unsigned int event = value == 0 ? 0 : 1;

                    activity[inputsIdx + batchInputOffset] = event;
                    firingRate[inputsIdx + batchInputOffset] += event;
                    totalFiringRate[inputsIdx + batchInputOffset] += event;
                    outputsActivity[inputsIdx + batchInputOffset] += value;
                    totalOutputsActivity[inputsIdx + batchInputOffset] += value;
            }
        }
    }
}



void N2D2::cudaUpdateMetrics(float * inputs,
                            int * activity,
                            long long unsigned int * firingRate,
                            long long unsigned int * totalFiringRate,
                            long long int * outputsActivity,
                            long long int * totalOutputsActivity,
                            unsigned int inputsDimX,
                            unsigned int inputsDimY,
                            unsigned int inputsDimZ,
                            unsigned int batchSize,
                            unsigned int maxNbThreads,
                            unsigned int warpSize)
{
    const unsigned int groupSize = inputsDimX * inputsDimY < maxNbThreads ?
        inputsDimX * inputsDimY : maxNbThreads;

    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)inputsDimX);

    const unsigned int groupX = min(warpSize, reqWidth);

    const dim3 blocksPerGrid = {inputsDimZ, 1, batchSize};
    const dim3 threadsPerBlocks = {groupX, groupSize/groupX, 1};

    cudaUpdateMetrics_kernel <<<blocksPerGrid, threadsPerBlocks>>> (
                inputs,
                activity,
                firingRate,
                totalFiringRate,
                outputsActivity,
                totalOutputsActivity,
                inputsDimX,
                inputsDimY,
                inputsDimZ);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

long long unsigned int N2D2::cudaIntegrateFiringRate(long long unsigned int * firingRate, unsigned int size)
{
    thrust::device_ptr<long long unsigned int> thrustPtr(firingRate);
    return thrust::reduce(thrustPtr, 
            thrustPtr + size, (long long unsigned int) 0, 
            thrust::plus<long long unsigned int>()); 
}

long long int N2D2::cudaIntegrateOutputsActivity(long long int * outputsActivity, unsigned int size)
{
    thrust::device_ptr<long long int> thrustPtr(outputsActivity);
    return thrust::reduce(thrustPtr, 
            thrustPtr + size, (long long int) 0, 
            thrust::plus<long long int>()); 
}
