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

#ifndef N2D2_TARGETBIASCELL_FRAME_CUDA_KERNELS_H
#define N2D2_TARGETBIASCELL_FRAME_CUDA_KERNELS_H

#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaUtils.hpp"
#include "CublasUtils.hpp"
#include "third_party/half.hpp"

namespace N2D2 {
template <class T>
void cudaTargetBiasPropagate(const cudaDeviceProp& deviceProp,
                       const T bias,
                       const T* inputs,
                       const T* diffInputs,
                       T* outputs,
                       unsigned int channelsHeight,
                       unsigned int channelsWidth,
                       unsigned int nbChannels,
                       unsigned int batchSize);
}

#endif // N2D2_TARGETBIASCELL_FRAME_CUDA_KERNELS_H
