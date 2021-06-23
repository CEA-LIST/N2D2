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

#ifndef N2D2_RESIZECELL_FRAME_CUDA_KERNELS_H
#define N2D2_RESIZECELL_FRAME_CUDA_KERNELS_H

#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaUtils.hpp"
namespace N2D2 {
// Forward
void cudaSResizeFWBilinearTF(const cudaDeviceProp& deviceProp, 
                           unsigned int outputSizeX,
                           unsigned int outputSizeY,
                           unsigned int nbChannels,
                           unsigned int batchSize,
                           unsigned int inputSizeX,
                           unsigned int inputSizeY,
                           unsigned int* yLowIdx,
                           unsigned int* yHighIdx,
                           float* yInter,
                           unsigned int* xLowIdx,
                           unsigned int* xHighIdx,
                           float* xInter,
                           const float* input,
                           float* outputs);

void cudaSResizeBWBilinearTF(const cudaDeviceProp& deviceProp, 
                               unsigned int outputSizeX,
                               unsigned int outputSizeY,
                               unsigned int nbChannels,
                               unsigned int batchSize,
                               unsigned int inputSizeX,
                               unsigned int inputSizeY,
                               const float scaleX,
                               const float scaleY,
                               const float* input,
                               float* outputs);

void cudaSResizeFWNearestNeighbor(const cudaDeviceProp& deviceProp, const float* input, size_t inputSizeX, size_t inputSizeY,
                                  float* output, size_t outputSizeX, size_t outputSizeY,
                                  size_t nbChannels, size_t batchSize);

void cudaSResizeBWNearestNeighbor(const cudaDeviceProp& deviceProp, const float* input, size_t inputSizeX, size_t inputSizeY,
                                  float* output, size_t outputSizeX, size_t outputSizeY,
                                  size_t nbChannels, size_t batchSize);
}

#endif // N2D2_RESIZECELL_FRAME_CUDA_KERNELS_H
