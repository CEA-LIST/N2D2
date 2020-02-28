/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#ifndef N2D2_NORMALIZECELL_FRAME_CUDA_KERNELS_H
#define N2D2_NORMALIZECELL_FRAME_CUDA_KERNELS_H

#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "CudaUtils.hpp"
#include "third_party/half.hpp"

namespace N2D2 {
/******************** Forward ***************************/
//Half
void cudaHNormalizeL2Forward(const cudaDeviceProp& deviceProp,
                             half_float::half alpha,
                             half_float::half* inputs,
                             unsigned int nbChannels,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int batchSize,
                             half_float::half beta,
                             half_float::half* outputs,
                             half_float::half* normData,
                             unsigned int nbOutputs,
                             unsigned int outputsHeight,
                             unsigned int outputsWidth);
//Float
void cudaSNormalizeL2Forward(const cudaDeviceProp& deviceProp,
                             const float alpha,
                             float* inputs,
                             unsigned int nbChannels,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int batchSize,
                             const float beta,
                             float* outputs,
                             float* normData,
                             unsigned int nbOutputs,
                             unsigned int outputsHeight,
                             unsigned int outputsWidth);
//Double
void cudaDNormalizeL2Forward(const cudaDeviceProp& deviceProp,
                             const double alpha,
                             double* inputs,
                             unsigned int nbChannels,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int batchSize,
                             const double beta,
                             double* outputs,
                             double* normData,
                             unsigned int nbOutputs,
                             unsigned int outputsHeight,
                             unsigned int outputsWidth);
/*********************************************************/

/******************** Backward ***************************/
//Half
void cudaHNormalizeL2Backward(const cudaDeviceProp& deviceProp,
                              half_float::half alpha,
                              half_float::half* outputs,
                              half_float::half* normData,
                              half_float::half* diffInputs,
                              unsigned int nbOutputs,
                              unsigned int outputsHeight,
                              unsigned int outputsWidth,
                              unsigned int batchSize,
                              half_float::half beta,
                              half_float::half* diffOutputs,
                              unsigned int nbChannels,
                              unsigned int channelsHeight,
                              unsigned int channelsWidth);
//Float
void cudaSNormalizeL2Backward(const cudaDeviceProp& deviceProp,
                              const float alpha,
                              float* outputs,
                              float* normData,
                              float* diffInputs,
                              unsigned int nbOutputs,
                              unsigned int outputsHeight,
                              unsigned int outputsWidth,
                              unsigned int batchSize,
                              const float beta,
                              float* diffOutputs,
                              unsigned int nbChannels,
                              unsigned int channelsHeight,
                              unsigned int channelsWidth);
//Double
void cudaDNormalizeL2Backward(const cudaDeviceProp& deviceProp,
                              const double alpha,
                              double* outputs,
                              double* normData,
                              double* diffInputs,
                              unsigned int nbOutputs,
                              unsigned int outputsHeight,
                              unsigned int outputsWidth,
                              unsigned int batchSize,
                              const double beta,
                              double* diffOutputs,
                              unsigned int nbChannels,
                              unsigned int channelsHeight,
                              unsigned int channelsWidth);
/*********************************************************/
}

#endif // N2D2_NORMALIZECELL_FRAME_CUDA_KERNELS_H
