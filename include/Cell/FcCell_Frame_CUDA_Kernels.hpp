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

#ifndef N2D2_FCCELL_FRAME_CUDA_KERNELS_H
#define N2D2_FCCELL_FRAME_CUDA_KERNELS_H

#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "CudaUtils.hpp"
#include "third_party/half.hpp"

namespace N2D2 {
template <class T>
void cudaFcWeightsSumSq(const cudaDeviceProp& deviceProp,
                         T* weights,
                         T* weightsNorm,
                         unsigned int nbChannels,
                         unsigned int nbOutputs);

template <class T>
void cudaFcWeightsSqrt(const cudaDeviceProp& deviceProp,
                        T* weightsNorm,
                        unsigned int nbOutputs,
                        T epsilon);

template <class T>
void cudaFcWeightsNormalize(const cudaDeviceProp& deviceProp,
                             T* weights,
                             T* weightsNorm,
                             unsigned int nbChannels,
                             unsigned int nbOutputs);

}

#endif // N2D2_FCCELL_FRAME_CUDA_KERNELS_H
