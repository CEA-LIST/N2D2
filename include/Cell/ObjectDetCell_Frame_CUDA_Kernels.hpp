/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_OBJECTDETCELL_FRAME_CUDA_KERNELS_H
#define N2D2_OBJECTDETCELL_FRAME_CUDA_KERNELS_H
#include "CudaUtils.hpp"

#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

namespace N2D2 {
                                            
void cudaSReduceIndex(  const unsigned int inputSize,
                        const unsigned int inputBatchOffset,
                        const unsigned int outputBatchOffset,
                        const float* valueThreshold,
                        const float* inputs,
                        int* outputMap,
                        float* scores,
                        const dim3 blocksPerGrid,
                        const dim3 threadsPerBlock);
                        
void cudaS_SSD_output_gathering(    unsigned int batchSize,
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
                                    const dim3 threadsPerBlock);


int copy_if_INT32(  const int* inputs,
                    int* outputs, 
                    unsigned int nbElements);

int copy_if_FP32(  const float* inputs,
                    float* outputs, 
                    unsigned int nbElements);

void thrust_sort_keys_INT32(float* inputs,
                            int* keys,
                            unsigned int nbElements,
                            unsigned int offset,
                            bool ascending = false);

void thrust_gather_INT32(const int* keys,
    const float* inputs,
    float* outputs,
    unsigned int nbElements,
    unsigned int inputOffset,
    unsigned int outputOffset);
}

#endif // N2D2_OBJECTDETCELL_FRAME_CUDA_KERNELS_H
