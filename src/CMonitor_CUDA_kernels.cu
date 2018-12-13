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

#include <stdio.h>



static unsigned int nextDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        ++v;
    return v;
}

static unsigned int findPower(unsigned int value)
{
    unsigned int v = value;
    unsigned int factor = 1;
    while (factor < v)
        factor*=2;
    return factor;
}



__global__ void cudaUpdateActivity_kernel(char * inputs,
                                        char * activity,
                                        unsigned int * firingRate,
                                        unsigned int * exampleFiringRate,
                                        unsigned long long int * firstEventTime,
                                        unsigned long long int * lastEventTime,
                                        unsigned int inputsDimX,
                                        unsigned int inputsDimY,
                                        unsigned int inputsDimZ,
                                        unsigned int long long timestamp)
{
    const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

    // One batch per block z dimension
    const unsigned int batchInputOffset = blockIdx.z * inputSize;

    for (unsigned int channel = blockIdx.x; channel < inputsDimZ; channel += gridDim.x) {
        for (unsigned int y = threadIdx.y; y < inputsDimY; y += blockDim.y) {
            for (unsigned int x = threadIdx.x; x < inputsDimX; x += blockDim.x) {

                    const unsigned int inputsIdx =
                        x + y*inputsDimX + channel*inputsDimX*inputsDimY;
                    char act = inputs[inputsIdx + batchInputOffset];
                    int counter = int(act);//(act != 0 ? 1 : 0);

                    activity[inputsIdx + batchInputOffset] = act;
                    firingRate[inputsIdx + batchInputOffset] += counter;
                    exampleFiringRate[inputsIdx + batchInputOffset] += counter;
            }
        }
    }
}


__global__ void cudaUpdateFiringRate_kernel(unsigned int * firingRate,
                                        unsigned int * totalFiringRatePartial,
                                        unsigned int inputsDimX,
                                        unsigned int inputsDimY,
                                        unsigned int inputsDimZ)
{

    const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

    const unsigned int batchInputOffset = blockIdx.z * inputSize;

    const unsigned int blockOffset = blockIdx.x * blockDim.x;

    const unsigned int partialIdx = threadIdx.x + blockOffset;

    extern __shared__ unsigned int partialSum[];

    // Perform first level of reduction during initialization
    // This is more efficient since we need all threads to load data
    // but the partial sum will see only half of the threads active
    //partialSum[threadIdx.x] = firingRate[partialIdx + batchInputOffset] +
    //    firingRate[partialIdx + blockDim.x + batchInputOffset];

    partialSum[threadIdx.x] = 0;
    if (partialIdx < inputSize){
        partialSum[threadIdx.x] = firingRate[partialIdx + batchInputOffset];
    }

    __syncthreads();

    // Reduction over neurons
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset){
            partialSum[threadIdx.x] += partialSum[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        totalFiringRatePartial[blockIdx.x+gridDim.x*blockIdx.z] = partialSum[0];
    }


}

__global__ void cudaUpdateBatchFiringRate_kernel(unsigned int * firingRate,
                                        unsigned int * batchFiringRate,
                                        unsigned int inputsDimX,
                                        unsigned int inputsDimY,
                                        unsigned int inputsDimZ,
                                        unsigned int batchSize)
{

    const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

    for (unsigned int channel = blockIdx.x; channel < inputsDimZ; channel += gridDim.x){
        for (unsigned int sy = 0; sy < inputsDimY; sy+=blockDim.y){
            for (unsigned int sx = 0; sx < inputsDimX; sx+=blockDim.x) {
                const unsigned int inputsIdx =
                        channel*inputsDimX*inputsDimY + sy*inputsDimX + sx;

                unsigned int batchSum = 0;
                for(unsigned int batch=0; batch<batchSize; ++batch) {
                    const unsigned int batchInputOffset = batch * inputSize;
                    batchSum += firingRate[inputsIdx + batchInputOffset];
                }
                batchFiringRate[inputsIdx] = batchSum;
            }
        }
    }

}


/*
__global__ void cudaUpdateMostActive_kernel(unsigned int * exampleIds,
                                        unsigned int * exampleFiringRate,
                                        unsigned int * mostActiveId,
                                        unsigned int inputsDimX,
                                        unsigned int inputsDimY,
                                        unsigned int inputsDimZ)
{

    const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

    const unsigned int batchInputOffset = blockIdx.z * inputSize;

    const unsigned int blockOffset = blockIdx.x * blockDim.x;

    const unsigned int partialIdx = threadIdx.x + blockOffset;

    // TODO: Also used shared memory for firing rates to avoid global
    // memory accesses
    extern __shared__ unsigned int partialActiveIdx[];

    // TODO: Index 0 has a slight advantage here
    partialActiveIdx[threadIdx.x] = 0;
    if (partialIdx < inputSize){
        partialActiveIdx[threadIdx.x] = exampleIds[partialIdx + batchInputOffset];
    }

    __syncthreads();

    // Reduction over neurons
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset){
            if (exampleFiringRate[partialActiveIdx[threadIdx.x]] <
                exampleFiringRate[partialActiveIdx[threadIdx.x + offset]]) {
                    partialActiveIdx[threadIdx.x] =
                            partialActiveIdx[threadIdx.x + offset];
            }
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
       mostActiveId[blockIdx.x+gridDim.x*blockIdx.z] = partialActiveIdx[0];
    }

}*/

__global__ void cudaUpdateMostActive_kernel(unsigned int * exampleFiringRate,
                                        unsigned int * mostActiveId,
                                        unsigned int inputsDimX,
                                        unsigned int inputsDimY,
                                        unsigned int inputsDimZ)
{

    const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

    const unsigned int batchInputOffset = blockIdx.z * inputSize;

    extern __shared__ unsigned int partialActiveIdx[];

    // For case that threadIdx.x > inputSize
    partialActiveIdx[threadIdx.x] = 0;

    // TODO: Index 0 has a slight advantage here
    for (unsigned int i=threadIdx.x; i<inputSize; i+=blockDim.x) {
        partialActiveIdx[threadIdx.x] = threadIdx.x;
    }

    // Search for max ID in each thread
    for (unsigned int i=threadIdx.x; i<inputSize; i+=blockDim.x) {
        if (exampleFiringRate[i + batchInputOffset] >
            exampleFiringRate[partialActiveIdx[threadIdx.x] + batchInputOffset]) {
                partialActiveIdx[threadIdx.x] = i;
        }
    }

    __syncthreads();

    // Reduction over neurons
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset){
            if (exampleFiringRate[partialActiveIdx[threadIdx.x] + batchInputOffset] <
                exampleFiringRate[partialActiveIdx[threadIdx.x + offset] + batchInputOffset]) {
                    partialActiveIdx[threadIdx.x] =
                            partialActiveIdx[threadIdx.x + offset];
            }
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
       mostActiveId[blockIdx.z] = partialActiveIdx[0];
    }

}


void N2D2::cudaUpdateActivity(char * inputs,
                            char * activity,
                            unsigned int * firingRate,
                            unsigned int * exampleFiringRate,
                            unsigned long long int * firstEventTime,
                            unsigned long long int * lastEventTime,
                            unsigned int inputsDimX,
                            unsigned int inputsDimY,
                            unsigned int inputsDimZ,
                            unsigned long long int timestamp,
                            unsigned int batchSize,
                            unsigned int maxNbThreads,
                            unsigned int warpSize)
{
    const unsigned int groupSize = inputsDimX * inputsDimY < maxNbThreads ?
        inputsDimX * inputsDimY : maxNbThreads;

    const unsigned int groupX = min(warpSize,
        nextDivisor(groupSize, inputsDimX));

    const dim3 blocksPerGrid = {inputsDimZ, 1, batchSize};
    const dim3 threadsPerBlocks = {groupX, groupSize/groupX, 1};

    cudaUpdateActivity_kernel <<<blocksPerGrid, threadsPerBlocks>>> (
                inputs,
                activity,
                firingRate,
                exampleFiringRate,
                firstEventTime,
                lastEventTime,
                inputsDimX,
                inputsDimY,
                inputsDimZ,
                timestamp);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}


void N2D2::cudaUpdateFiringRate(unsigned int * firingRate,
                            unsigned int * totalFiringRate,
                            unsigned int inputsDimX,
                            unsigned int inputsDimY,
                            unsigned int inputsDimZ,
                            unsigned int batchSize,
                            unsigned int maxNbThreads,
                            unsigned int warpSize)
{
    unsigned int numElem = findPower(inputsDimX * inputsDimY * inputsDimZ);
    unsigned int nbBlocks = 1;
    while (numElem > maxNbThreads){
        numElem = numElem / 2;
        nbBlocks*=2;
    }


    const dim3 blocksPerGrid = {nbBlocks, 1, batchSize};
    const dim3 threadsPerBlocks = {numElem, 1, 1};
    size_t sharedSize = sizeof(unsigned int) * threadsPerBlocks.x;

    unsigned int * blockSum;
    cudaMallocManaged(&blockSum, blocksPerGrid.x * blocksPerGrid.z * sizeof(unsigned int));

    cudaUpdateFiringRate_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (
        firingRate,
        blockSum,
        inputsDimX,
        inputsDimY,
        inputsDimZ);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());


    cudaDeviceSynchronize();

    // Perform second sum over the partial sum in all blocks

    const dim3 blocksPerGrid2 = {1, 1, batchSize};
    const dim3 threadsPerBlocks2 = {nbBlocks, 1, 1};
    sharedSize = sizeof(unsigned int) * threadsPerBlocks2.x;

    cudaUpdateFiringRate_kernel <<< blocksPerGrid2, threadsPerBlocks2.x, sharedSize>>> (
        blockSum,
        totalFiringRate,
        nbBlocks,
        1,
        1);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());



    cudaFree(blockSum);

}

void N2D2::cudaUpdateBatchFiringRate(unsigned int * firingRate,
                                    unsigned int * batchFiringRate,
                                    unsigned int inputsDimX,
                                    unsigned int inputsDimY,
                                    unsigned int inputsDimZ,
                                    unsigned int batchSize,
                                    unsigned int maxNbThreads,
                                    unsigned int warpSize)
{
    const unsigned int groupSize = inputsDimX * inputsDimY < maxNbThreads ?
        inputsDimX * inputsDimY : maxNbThreads;

    const unsigned int groupX = min(warpSize,
        nextDivisor(groupSize, inputsDimX));

    const dim3 blocksPerGrid = {inputsDimZ, 1, 1};
    const dim3 threadsPerBlocks = {groupX, groupSize/groupX, 1};

    cudaUpdateBatchFiringRate_kernel <<<blocksPerGrid, threadsPerBlocks>>> (
        firingRate,
        batchFiringRate,
        inputsDimX,
        inputsDimY,
        inputsDimZ,
        batchSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}


/* //This does not work for batch size > 1!
void N2D2::cudaUpdateMostActive(unsigned int * exampleIds,
                            unsigned int * exampleFiringRate,
                            unsigned int * mostActiveId,
                            unsigned int inputsDimX,
                            unsigned int inputsDimY,
                            unsigned int inputsDimZ,
                            unsigned int batchSize,
                            unsigned int maxNbThreads,
                            unsigned int warpSize)
{
    unsigned int numElem = findPower(inputsDimX * inputsDimY * inputsDimZ);
    unsigned int nbBlocks = 1;
    while (numElem > maxNbThreads){
        numElem = numElem / 2;
        nbBlocks*=2;
    }

    // Distribute search for max in each batch over several blocks
    const dim3 blocksPerGrid = {nbBlocks, 1, batchSize};
    const dim3 threadsPerBlocks = {numElem, 1, 1};
    size_t sharedSize = sizeof(unsigned int) * threadsPerBlocks.x;

    unsigned int * blockActiveIds;
    cudaMallocManaged(&blockActiveIds,
                      blocksPerGrid.x * blocksPerGrid.z * sizeof(unsigned int));

    cudaUpdateMostActive_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (
        exampleIds,
        exampleFiringRate,
        blockActiveIds,
        inputsDimX,
        inputsDimY,
        inputsDimZ);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());


    // Perform second sum over the partial sum in all blocks

    const dim3 blocksPerGrid2 = {1, 1, batchSize};
    const dim3 threadsPerBlocks2 = {nbBlocks, 1, 1};
    sharedSize = sizeof(unsigned int) * threadsPerBlocks2.x;

    cudaUpdateMostActive_kernel <<< blocksPerGrid2, threadsPerBlocks2.x, sharedSize>>> (
        blockActiveIds,
        exampleFiringRate,
        mostActiveId,
        nbBlocks,
        1,
        1);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());


    cudaFree(blockActiveIds);


}*/

void N2D2::cudaUpdateMostActive(unsigned int * exampleFiringRate,
                            unsigned int * mostActiveId,
                            unsigned int inputsDimX,
                            unsigned int inputsDimY,
                            unsigned int inputsDimZ,
                            unsigned int batchSize,
                            unsigned int maxNbThreads,
                            unsigned int warpSize)
{
    unsigned int numElem = findPower(inputsDimX * inputsDimY * inputsDimZ);
    while (numElem > maxNbThreads){
        numElem = numElem / 2;
    }

    const dim3 blocksPerGrid = {1, 1, batchSize};
    const dim3 threadsPerBlocks = {numElem, 1, 1};
    size_t sharedSize = sizeof(unsigned int) * threadsPerBlocks.x;

    cudaUpdateMostActive_kernel <<<blocksPerGrid, threadsPerBlocks, sharedSize>>> (
        exampleFiringRate,
        mostActiveId,
        inputsDimX,
        inputsDimY,
        inputsDimZ);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}


