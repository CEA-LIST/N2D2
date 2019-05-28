/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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
#ifndef N2D2_TARGET_CUDA_KERNELS_H
#define N2D2_TARGET_CUDA_KERNELS_H

#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "CudaUtils.hpp"
#include "third_party/half.hpp"

namespace N2D2 {

void cudaGetEstimatedTarget(const unsigned int topN,
                            const unsigned int nbClass,
                            const unsigned int targetHeight,
                            const unsigned int targetWidth,
                            const unsigned int batchSize,
                            float threshold,
                            const float* input,
                            float* estimatedLabelsValueGPU,
                            int* estimatedLabelsGPU);

void cudaGetEstimatedLabel(const cudaDeviceProp& deviceProp,
                           const float* value,
                           unsigned int outputWidth,
                           unsigned int outputHeight,
                           unsigned int nbOutputs,
                           unsigned int batchPos,
                           unsigned int x0,
                           unsigned int x1,
                           unsigned int y0,
                           unsigned int y1,
                           float* bbLabels,
                           const int* mask,
                           int maskedLabel);
}

#endif // N2D2_TARGET_CUDA_KERNELS_H
