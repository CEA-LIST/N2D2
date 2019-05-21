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

#ifndef N2D2_POOLCELL_FRAME_CUDA_KERNELS_H
#define N2D2_POOLCELL_FRAME_CUDA_KERNELS_H

#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "CudaUtils.hpp"
#include "PoolCell_Frame_Kernels_struct.hpp"
#include "third_party/half.hpp"

namespace N2D2 {
/******************** Forward ***************************/
//Half
void cudaHPoolForwardAverage(const cudaDeviceProp& deviceProp,
                             half_float::half alpha,
                             half_float::half* inputs,
                             unsigned int nbChannels,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int batchSize,
                             const PoolCell_Frame_Kernels::Descriptor* desc,
                             half_float::half beta,
                             half_float::half* outputs,
                             unsigned int nbOutputs,
                             unsigned int outputsHeight,
                             unsigned int outputsWidth,
                             bool countIncludePadding = true,
                             char* maps = NULL);
//Float
void cudaSPoolForwardAverage(const cudaDeviceProp& deviceProp,
                             const float alpha,
                             float* inputs,
                             unsigned int nbChannels,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int batchSize,
                             const PoolCell_Frame_Kernels::Descriptor* desc,
                             const float beta,
                             float* outputs,
                             unsigned int nbOutputs,
                             unsigned int outputsHeight,
                             unsigned int outputsWidth,
                             bool countIncludePadding = true,
                             char* maps = NULL);
//Double
void cudaDPoolForwardAverage(const cudaDeviceProp& deviceProp,
                             const double alpha,
                             double* inputs,
                             unsigned int nbChannels,
                             unsigned int channelsHeight,
                             unsigned int channelsWidth,
                             unsigned int batchSize,
                             const PoolCell_Frame_Kernels::Descriptor* desc,
                             const double beta,
                             double* outputs,
                             unsigned int nbOutputs,
                             unsigned int outputsHeight,
                             unsigned int outputsWidth,
                             bool countIncludePadding = true,
                             char* maps = NULL);
//Half
void cudaHPoolForwardMax(const cudaDeviceProp& deviceProp,
                         half_float::half alpha,
                         half_float::half* inputs,
                         unsigned int nbChannels,
                         unsigned int channelsHeight,
                         unsigned int channelsWidth,
                         unsigned int batchSize,
                         const PoolCell_Frame_Kernels::Descriptor* desc,
                         half_float::half beta,
                         half_float::half* outputs,
                         unsigned int nbOutputs,
                         unsigned int outputsHeight,
                         unsigned int outputsWidth,
                         PoolCell_Frame_Kernels::ArgMax* argMax,
                         bool useArgMax = false,
                         char* maps = NULL);
//Float
void cudaSPoolForwardMax(const cudaDeviceProp& deviceProp,
                         const float alpha,
                         float* inputs,
                         unsigned int nbChannels,
                         unsigned int channelsHeight,
                         unsigned int channelsWidth,
                         unsigned int batchSize,
                         const PoolCell_Frame_Kernels::Descriptor* desc,
                         const float beta,
                         float* outputs,
                         unsigned int nbOutputs,
                         unsigned int outputsHeight,
                         unsigned int outputsWidth,
                         PoolCell_Frame_Kernels::ArgMax* argMax,
                         bool useArgMax = false,
                         char* maps = NULL);
//Double
void cudaDPoolForwardMax(const cudaDeviceProp& deviceProp,
                         const double alpha,
                         double* inputs,
                         unsigned int nbChannels,
                         unsigned int channelsHeight,
                         unsigned int channelsWidth,
                         unsigned int batchSize,
                         const PoolCell_Frame_Kernels::Descriptor* desc,
                         const double beta,
                         double* outputs,
                         unsigned int nbOutputs,
                         unsigned int outputsHeight,
                         unsigned int outputsWidth,
                         PoolCell_Frame_Kernels::ArgMax* argMax,
                         bool useArgMax = false,
                         char* maps = NULL);
/*********************************************************/

/******************** Backward ***************************/
//Half
void cudaHPoolBackwardAverage(const cudaDeviceProp& deviceProp,
                              half_float::half alpha,
                              half_float::half* diffInputs,
                              unsigned int nbOutputs,
                              unsigned int outputsHeight,
                              unsigned int outputsWidth,
                              unsigned int batchSize,
                              const PoolCell_Frame_Kernels::Descriptor* desc,
                              half_float::half beta,
                              half_float::half* diffOutputs,
                              unsigned int nbChannels,
                              unsigned int channelsHeight,
                              unsigned int channelsWidth,
                              bool countIncludePadding = true,
                              char* maps = NULL);
//Float
void cudaSPoolBackwardAverage(const cudaDeviceProp& deviceProp,
                              const float alpha,
                              float* diffInputs,
                              unsigned int nbOutputs,
                              unsigned int outputsHeight,
                              unsigned int outputsWidth,
                              unsigned int batchSize,
                              const PoolCell_Frame_Kernels::Descriptor* desc,
                              const float beta,
                              float* diffOutputs,
                              unsigned int nbChannels,
                              unsigned int channelsHeight,
                              unsigned int channelsWidth,
                              bool countIncludePadding = true,
                              char* maps = NULL);
//Double
void cudaDPoolBackwardAverage(const cudaDeviceProp& deviceProp,
                              const double alpha,
                              double* diffInputs,
                              unsigned int nbOutputs,
                              unsigned int outputsHeight,
                              unsigned int outputsWidth,
                              unsigned int batchSize,
                              const PoolCell_Frame_Kernels::Descriptor* desc,
                              const double beta,
                              double* diffOutputs,
                              unsigned int nbChannels,
                              unsigned int channelsHeight,
                              unsigned int channelsWidth,
                              bool countIncludePadding = true,
                              char* maps = NULL);
//Half
void cudaHPoolBackwardMax(const cudaDeviceProp& deviceProp,
                          half_float::half alpha,
                          half_float::half* diffInputs,
                          unsigned int nbOutputs,
                          unsigned int outputsHeight,
                          unsigned int outputsWidth,
                          unsigned int batchSize,
                          const PoolCell_Frame_Kernels::Descriptor* desc,
                          half_float::half beta,
                          half_float::half* diffOutputs,
                          unsigned int nbChannels,
                          unsigned int channelsHeight,
                          unsigned int channelsWidth,
                          PoolCell_Frame_Kernels::ArgMax* argMax,
                          char* maps = NULL);
//Float
void cudaSPoolBackwardMax(const cudaDeviceProp& deviceProp,
                          const float alpha,
                          float* diffInputs,
                          unsigned int nbOutputs,
                          unsigned int outputsHeight,
                          unsigned int outputsWidth,
                          unsigned int batchSize,
                          const PoolCell_Frame_Kernels::Descriptor* desc,
                          const float beta,
                          float* diffOutputs,
                          unsigned int nbChannels,
                          unsigned int channelsHeight,
                          unsigned int channelsWidth,
                          PoolCell_Frame_Kernels::ArgMax* argMax,
                          char* maps = NULL);
//Double
void cudaDPoolBackwardMax(const cudaDeviceProp& deviceProp,
                          const double alpha,
                          double* diffInputs,
                          unsigned int nbOutputs,
                          unsigned int outputsHeight,
                          unsigned int outputsWidth,
                          unsigned int batchSize,
                          const PoolCell_Frame_Kernels::Descriptor* desc,
                          const double beta,
                          double* diffOutputs,
                          unsigned int nbChannels,
                          unsigned int channelsHeight,
                          unsigned int channelsWidth,
                          PoolCell_Frame_Kernels::ArgMax* argMax,
                          char* maps = NULL);

}

#endif // N2D2_POOLCELL_FRAME_CUDA_KERNELS_H
