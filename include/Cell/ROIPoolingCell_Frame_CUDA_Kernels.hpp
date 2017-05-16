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

#ifndef N2D2_ROIPOOLINGCELL_FRAME_CUDA_KERNELS_H
#define N2D2_ROIPOOLINGCELL_FRAME_CUDA_KERNELS_H

#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "PoolCell_Frame_Kernels_struct.hpp"

namespace N2D2 {
// Forward
void cudaSROIPoolingForwardAverage(const float alpha,
                                   float* proposals,
                                   unsigned int nbProposals,
                                   unsigned int inputSizeY,
                                   unsigned int inputSizeX,
                                   float* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   const float beta,
                                   float* outputs,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth);

void cudaSROIPoolingForwardMax(const float alpha,
                               float* proposals,
                               unsigned int nbProposals,
                               unsigned int inputSizeY,
                               unsigned int inputSizeX,
                               float* inputs,
                               unsigned int nbChannels,
                               unsigned int channelsHeight,
                               unsigned int channelsWidth,
                               unsigned int batchSize,
                               const float beta,
                               float* outputs,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               PoolCell_Frame_Kernels::ArgMax* argMax);

// Backward
void cudaSROIPoolingBackwardAverage(const float alpha,
                                    float* proposals,
                                    unsigned int nbProposals,
                                    unsigned int inputSizeY,
                                    unsigned int inputSizeX,
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

void cudaSROIPoolingBackwardMax(const float alpha,
                                float* proposals,
                                unsigned int nbProposals,
                                unsigned int inputSizeY,
                                unsigned int inputSizeX,
                                float* diffInputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                unsigned int batchSize,
                                const float beta,
                                float* diffOutputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                PoolCell_Frame_Kernels::ArgMax* argMax);
}

#endif // N2D2_ROIPOOLINGCELL_FRAME_CUDA_KERNELS_H
