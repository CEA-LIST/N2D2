
/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND(david.briand@cea.fr)

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

#include "Cell/ObjectDetCell_Frame_CUDA_Kernels.hpp"

/**Reduceindex Kernel create a map index where value is superior to Threshold**/
__global__ void cudaSReduceIndex_kernel(  const unsigned int inputSize,
                                          const unsigned int inputBatchOffset,
                                          const unsigned int outputBatchOffset,
                                          const float valueThreshold,
                                          const float* inputs,
                                          int* outputMap)
{
    const int batchPos = blockIdx.z;
    const int clsPos = blockIdx.y;

    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    const int inputIndex = index 
                            + inputSize*blockIdx.y
                            + batchPos*inputBatchOffset;

    const int outputIndex = index
                            + inputSize*blockIdx.y
                            + batchPos*outputBatchOffset;

    if(index < inputSize)
    {
        float value = inputs[inputIndex];      

        if(value >= valueThreshold)
            outputMap[outputIndex] = index;
        else
            outputMap[outputIndex] = -1;
        
    }  
}

void N2D2::cudaSReduceIndex(  const unsigned int inputSize,
                              const unsigned int inputBatchOffset,
                              const unsigned int outputBatchOffset,
                              const float valueThreshold,
                              const float* inputs,
                              int* outputMap,
                              const dim3 blocksPerGrid,
                              const dim3 threadsPerBlock)
{

    cudaSReduceIndex_kernel<<<blocksPerGrid, threadsPerBlock>>>( inputSize,
                                                                 inputBatchOffset,
                                                                 outputBatchOffset,
                                                                 valueThreshold, 
                                                                 inputs,
                                                                 outputMap);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}

int N2D2::copy_if(const int* inputs,
                   int* outputs, 
                   unsigned int nbElements)
{
    const thrust::device_ptr<const int> thrust_data_inputs(inputs);
    const thrust::device_ptr<int> thrust_data_outputs(outputs);
    thrust::copy_if(  thrust_data_inputs,
                                thrust_data_inputs + nbElements,
                                thrust_data_outputs ,
                                thrust::placeholders::_1 > -1);  
    return 1;
}

void N2D2::thrust_gather(const int* keys, 
                         const float* inputs,
                         float* outputs, 
                         unsigned int nbElements,
                         unsigned int inputOffset,
                         unsigned int outputOffset)
{
    const thrust::device_ptr<const float> thrust_data_inputs(inputs);
    const thrust::device_ptr<const int> thrust_keys(keys);
    const thrust::device_ptr<float> thrust_data_outputs(outputs);
    thrust::gather(thrust_keys + inputOffset,
                   thrust_keys + inputOffset + nbElements,
                   thrust_data_inputs + inputOffset,
                   thrust_data_outputs + outputOffset);    
}
