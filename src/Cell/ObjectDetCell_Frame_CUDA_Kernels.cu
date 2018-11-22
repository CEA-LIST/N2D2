
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
                                          const float* valueThreshold,
                                          const float* inputs,
                                          int* outputMap,
                                          float* scores)
{
    const int batchPos = blockIdx.z;
    const int clsPos = blockIdx.y;

    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    const int inputIndex = index
                            + inputSize*clsPos
                            + batchPos*inputBatchOffset;

    const int outputIndex = index
                            + inputSize*clsPos
                            + batchPos*outputBatchOffset;

    if(index < inputSize)
    {

        float value = inputs[inputIndex];

        if(value >= valueThreshold[clsPos])
        {
            outputMap[outputIndex] = index;
            scores[outputIndex] = value;
        }
        else
        {
            outputMap[outputIndex] = -1;
            scores[outputIndex] = -1.0;
        }

    }
}

void N2D2::cudaSReduceIndex(  const unsigned int inputSize,
                              const unsigned int inputBatchOffset,
                              const unsigned int outputBatchOffset,
                              const float* valueThreshold,
                              const float* inputs,
                              int* outputMap,
                              float* scores,
                              const dim3 blocksPerGrid,
                              const dim3 threadsPerBlock)
{

    cudaSReduceIndex_kernel<<<blocksPerGrid, threadsPerBlock>>>( inputSize,
                                                                 inputBatchOffset,
                                                                 outputBatchOffset,
                                                                 valueThreshold,
                                                                 inputs,
                                                                 outputMap,
                                                                 scores);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}

__global__ void cudaS_ssdToOutput_kernels(  unsigned int batchSize,
                                            unsigned int nbClass,
                                            unsigned int nbAnchors,
                                            unsigned int channelWidth,
                                            unsigned int channelHeight,
                                            unsigned int nbProposals,
                                            unsigned int* nbValidROIs,
                                            unsigned int cls,
                                            unsigned int totalParts,
                                            unsigned int totalTemplates,
                                            unsigned int maxParts,
                                            unsigned int maxTemplates,
                                            unsigned int cumulParts,
                                            unsigned int cumulTemplates,
                                            unsigned int nbParts,
                                            unsigned int nbTemplates,
                                            float xRatio,
                                            float yRatio,
                                            float xOutputRatio,
                                            float yOutputRatio,
                                            const float* roi_bbox,
                                            const float* roi_anchors,
                                            const float* anchors,
                                            const float* inputs_parts,
                                            const float* inputs_templates,
                                            float* outputs)
{
    const int batchPos = blockIdx.z;
    const int proposal = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;
    const int ptIdx = blockIdx.y;
    
    const int nbDetectedObject  = (int) nbValidROIs[batchPos];
    const int nbIdx = 6;
    if(proposal < nbProposals)
    {
        const unsigned int n = proposal + cls*nbProposals + batchPos*nbProposals*nbClass;

        if(proposal < nbDetectedObject)
        {
            if(ptIdx == 0)
            {
                outputs[0 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[0 + 5*proposal + batchPos*nbProposals*5];
                outputs[1 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[1 + 5*proposal + batchPos*nbProposals*5];
                outputs[2 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[2 + 5*proposal + batchPos*nbProposals*5];
                outputs[3 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[3 + 5*proposal + batchPos*nbProposals*5];
                outputs[4 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[4 + 5*proposal + batchPos*nbProposals*5];
                outputs[5 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = (float) cls;
            }
            /*if(ptIdx < nbParts)
            {
                outputs[ptIdx*2 + 0 + 5 + n*(5 + maxParts*2 + maxTemplates*3) ] = 0.0;
                outputs[ptIdx*2 + 1 + 5 + n*(5 + maxParts*2 + maxTemplates*3) ] = 0.0;
            }

            if(ptIdx < nbTemplates)
            {
                outputs[ptIdx*3 + maxParts*2 + 0 + 5 + n*(5 + maxParts*2 + maxTemplates*3) ] = 0.0;
                outputs[ptIdx*3 + maxParts*2 + 1 + 5 + n*(5 + maxParts*2 + maxTemplates*3) ] = 0.0;
                outputs[ptIdx*3 + maxParts*2 + 2 + 5 + n*(5 + maxParts*2 + maxTemplates*3) ] = 0.0;
            }*/

            const unsigned int xa   = roi_anchors[0 + 5*proposal + batchPos*nbProposals*5];
            const unsigned int ya   = roi_anchors[1 + 5*proposal + batchPos*nbProposals*5];
            const unsigned int k    = roi_anchors[2 + 5*proposal + batchPos*nbProposals*5];
           
            if(ptIdx < nbParts)
            {
                const int yIdx = xa 
                                + ya*channelWidth 
                                + (k*nbParts*2 + cumulParts + ptIdx*2)*channelHeight*channelWidth
                                + batchPos*channelHeight*channelWidth*nbAnchors*2*totalParts;
                const int xIdx = xa 
                                + ya*channelWidth 
                                + (k*nbParts*2 + cumulParts + ptIdx*2 + 1)*channelHeight*channelWidth
                                + batchPos*channelHeight*channelWidth*nbAnchors*2*totalParts;


                const float partY = inputs_parts[yIdx];
                const float partX = inputs_parts[xIdx];

                const int xa0 = (int)(anchors[k*4] + xa * xRatio);
                const int ya0 = (int)(anchors[k*4 + 1] + ya * yRatio);
                const int xa1 = (int)(anchors[k*4 + 2] + xa * xRatio);
                const int ya1 = (int)(anchors[k*4 + 3] + ya * yRatio);

                // Anchors width and height
                const int wa = xa1 - xa0;
                const int ha = ya1 - ya0;

                // Anchor center coordinates (xac, yac)
                const float xac = xa0 + wa / 2.0;
                const float yac = ya0 + ha / 2.0;
                const float predPartY = ((partY) * ha + yac)*yOutputRatio ;
                const float predPartX = ((partX) * wa + xac)*xOutputRatio ;

                outputs[ptIdx*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = predPartY;
                outputs[ptIdx*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = predPartX;

            }
            else if(ptIdx < maxParts)
            {
                    outputs[ptIdx*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                    outputs[ptIdx*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
            }
            
            ///for(unsigned int t = 0; t < nbTemplates; ++t)
            if(ptIdx < nbTemplates)
            {
                const int yIdx = xa 
                                + ya*channelWidth 
                                + (k*nbTemplates*3 + cumulTemplates + ptIdx*3)*channelHeight*channelWidth
                                + batchPos*channelHeight*channelWidth*nbAnchors*3*totalTemplates;
                const int xIdx = xa 
                                + ya*channelWidth 
                                + (k*nbTemplates*3 + cumulTemplates + ptIdx*3 + 1)*channelHeight*channelWidth
                                + batchPos*channelHeight*channelWidth*nbAnchors*3*totalTemplates;
                const int zIdx = xa 
                                + ya*channelWidth 
                                + (k*nbTemplates*3 + cumulTemplates + ptIdx*3 + 2)*channelHeight*channelWidth
                                + batchPos*channelHeight*channelWidth*nbAnchors*3*totalTemplates;


                const float templateY = expf(inputs_templates[yIdx]);
                const float templateX = expf(inputs_templates[xIdx]);
                const float templateZ = expf(inputs_templates[zIdx]);

                outputs[ptIdx*3 + maxParts*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = templateY;
                outputs[ptIdx*3 + maxParts*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = templateX;
                outputs[ptIdx*3 + maxParts*2 + 2 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = templateZ;

            }
            else if(ptIdx < maxTemplates)
            {
                    outputs[ptIdx*3 + maxParts*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                    outputs[ptIdx*3 + maxParts*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                    outputs[ptIdx*3 + maxParts*2 + 2 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
            }
        }
        else
        {
            outputs[0 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            outputs[1 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            outputs[2 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            outputs[3 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            outputs[4 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            //for(unsigned int p = 0; p < nbParts; ++p)
            if(ptIdx < maxParts)
            {
                outputs[ptIdx*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                outputs[ptIdx*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
            }

            //for(unsigned int t = 0;t < nbTemplates; ++t)
            if(ptIdx < maxTemplates)
            {
                outputs[ptIdx*3 + maxParts*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                outputs[ptIdx*3 + maxParts*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                outputs[ptIdx*3 + maxParts*2 + 2 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
            }
        }
    }
}



void N2D2::cudaS_SSD_output_gathering( unsigned int batchSize,
                                        unsigned int nbClass,
                                        unsigned int nbAnchors,
                                        unsigned int channelWidth,
                                        unsigned int channelHeight,
                                        unsigned int nbProposals,
                                        unsigned int* nbValidROIs,
                                        unsigned int cls,
                                        unsigned int totalParts,
                                        unsigned int totalTemplates,
                                        unsigned int maxParts,
                                        unsigned int maxTemplates,
                                        unsigned int cumulParts,
                                        unsigned int cumulTemplates,
                                        unsigned int nbParts,
                                        unsigned int nbTemplates,
                                        float xRatio,
                                        float yRatio,
                                        float xOutputRatio,
                                        float yOutputRatio,
                                        const float* roi_bbox,
                                        const float* roi_anchors,
                                        const float* anchors,
                                        const float* inputs_parts,
                                        const float* inputs_templates,
                                        float* outputs,
                                        const dim3 blocksPerGrid,
                                        const dim3 threadsPerBlock)
{

    cudaS_ssdToOutput_kernels<<<blocksPerGrid, threadsPerBlock>>>( batchSize,
                                                                nbClass,
                                                                nbAnchors,
                                                                channelWidth,
                                                                channelHeight,
                                                                nbProposals,
                                                                nbValidROIs,
                                                                cls,
                                                                totalParts,
                                                                totalTemplates,
                                                                maxParts,
                                                                maxTemplates,
                                                                cumulParts,
                                                                cumulTemplates,
                                                                nbParts,
                                                                nbTemplates,
                                                                xRatio,
                                                                yRatio,
                                                                xOutputRatio,
                                                                yOutputRatio,
                                                                roi_bbox,
                                                                roi_anchors,
                                                                anchors,
                                                                inputs_parts,
                                                                inputs_templates,
                                                                outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}



int N2D2::copy_if_INT32(  const int* inputs,
                          int* outputs,
                          unsigned int nbElements)
{
    const thrust::device_ptr<const int> thrust_data_inputs(inputs);
    const thrust::device_ptr<int> thrust_data_outputs(outputs);
    thrust::device_ptr<int> return_ptr = thrust::copy_if(   thrust_data_inputs,
                                                            thrust_data_inputs + nbElements,
                                                            thrust_data_outputs ,
                                                            thrust::placeholders::_1 > -1);

    int nbCpyElements = (int) (return_ptr - thrust_data_outputs);

    return nbCpyElements;
}


int N2D2::copy_if_FP32(  const float* inputs,
                         float* outputs,
                         unsigned int nbElements)
{
    const thrust::device_ptr<const float> thrust_data_inputs(inputs);
    const thrust::device_ptr<float> thrust_data_outputs(outputs);

    thrust::device_ptr<float> return_ptr =  thrust::copy_if(   thrust_data_inputs,
                                                                thrust_data_inputs + nbElements,
                                                                thrust_data_outputs ,
                                                                thrust::placeholders::_1 > -1);
    int nbCpyElements = (int) (return_ptr - thrust_data_outputs);

    return nbCpyElements;
}

void N2D2::thrust_gather_INT32( const int* keys,
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

void N2D2::thrust_sort_keys_INT32(float* inputs, int* keys, unsigned int nbElements,  unsigned int offset)
{

    const thrust::device_ptr<float> thrust_data(inputs);
    const thrust::device_ptr<int> thrust_keys(keys);

    thrust::stable_sort_by_key( thrust_data + offset,
                                thrust_data + offset + nbElements,
                                thrust_keys + offset,
                                thrust::greater<float>());

}

