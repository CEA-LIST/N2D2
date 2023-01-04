/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#ifndef N2D2_PRUNEQUANTIZER_FRAME_CUDA_KERNELS_H
#define N2D2_PRUNEQUANTIZER_FRAME_CUDA_KERNELS_H

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>


#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include "CudaUtils.hpp"
#include "third_party/half.hpp"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

namespace N2D2 {

namespace PruneQuantizer_Frame_CUDA_Kernels {

    void apply_pruning_with_masks_H(half_float::half* data,
                                    half_float::half* dataPruned,
                                    unsigned int* masks,
                                    unsigned int size);

    void apply_pruning_with_masks_F(float* data,
                                    float* dataPruned,
                                    unsigned int* masks,
                                    unsigned int size);

    void apply_pruning_with_masks_D(double* data,
                                    double* dataPruned,
                                    unsigned int* masks,
                                    unsigned int size);

}

}



#endif  // N2D2_PRUNEQUANTIZER_FRAME_CUDA_KERNELS_H