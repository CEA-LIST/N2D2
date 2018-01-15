/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_RP_FRAME_CUDA_KERNELS_H
#define N2D2_RP_FRAME_CUDA_KERNELS_H

#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

namespace N2D2 {
                                            
void cudaSSplitIndexes(  unsigned int inputSizeX,
                         unsigned int inputSizeY,
                         unsigned int nbAnchors,
                         unsigned int batchSize,
                         unsigned int nbBlocks,
                         const float* inputs,
                         float* values,
                         int* indexI,
                         int* indexJ,
                         int* indexK,
                         int* indexB,
                         unsigned int* map,
                         float minWidth,
                         float minHeight,
                         unsigned int scoreIndex);

void cudaSnms( unsigned int inputSizeX,
               unsigned int inputSizeY,
               unsigned int nbAnchors,
               unsigned int batchSize,
               const float* inputs,
               const unsigned int inputOffset,
               int* i,
               int* j,
               int* k,
               int* b,
               const unsigned int indexOffset,
               unsigned long long* mask,
               const unsigned int outputOffset,
               const float nms_iou_thresh,
               const unsigned int max_nbBoxes,
               const dim3 threadsPerBlock,
               const dim3 blocksPerGrid);
   
void cudaSGatherRP( unsigned int inputSizeX,
                    unsigned int inputSizeY,
                    unsigned int nbAnchors,
                    unsigned int batchSize,
                    const float* inputs,
                    const int* i,
                    const int* j,
                    const int* k,
                    const int* b,
                    const int* mask,
                    float* outputs,
                    int* anchors,
                    unsigned int topN,
                    const unsigned int nbProposals,
                    const unsigned int nbBlocks);   

void thrust_sort(float* inputs, unsigned int nbElements);

void thrust_sort_keys(float* inputs, 
                      unsigned int* keys,
                      unsigned int nbElements,
                      unsigned int offset = 0);

void thrust_gather(const unsigned int* keys, 
                   const int* inputs,
                   int* outputs, 
                   unsigned int nbElements,
                   unsigned int inputOffset = 0,
                   unsigned int outputOffset = 0);

                            
}

#endif // N2D2_ANCHORCELL_FRAME_CUDA_KERNELS_H

