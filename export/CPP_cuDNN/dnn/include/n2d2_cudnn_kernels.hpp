/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#ifndef DEEPNET_CUDNN_KERNELS_H
#define DEEPNET_CUDNN_KERNELS_H

#include "../../include/typedefs.h"
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

void cudaSFMPPropagate(const DATA_T* inputs,
                       unsigned int* gridX,
                       unsigned int* gridY,
                       DATA_T* outputs,
                       unsigned int nbChannels,
                       unsigned int channelsHeight,
                       unsigned int channelsWidth,
                       unsigned int nbOutputs,
                       unsigned int outputsHeight,
                       unsigned int outputsWidth,
                       unsigned int batchSize,
                       bool overlapping);

void cudaSBNPropagate(const DATA_T* inputs,
                      float* bias,
                      float* variance,
                      float* mean,
                      float* scale,
                      float epsilon,
                      DATA_T* outputs,
                      unsigned int nbChannels,
                      unsigned int channelsHeight,
                      unsigned int channelsWidth,
                      unsigned int batchSize);

#endif // DEEPNET_FMPCELL_FRAME_CUDA_KERNELS_H
