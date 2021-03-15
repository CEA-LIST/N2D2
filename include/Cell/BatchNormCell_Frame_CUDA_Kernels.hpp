/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Vincent TEMPLIER (vincent.templier@cea.fr)

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

#ifndef N2D2_BATCHNORMCELL_FRAME_CUDA_KERNELS_H
#define N2D2_BATCHNORMCELL_FRAME_CUDA_KERNELS_H

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "third_party/half.hpp"

namespace N2D2 {


template <typename T>
void thrust_div(T* /*srcData*/, size_t /*size*/, int /*value*/) {}

template <> void thrust_div(half_float::half* srcData,
                            size_t size,
                            int value);
template <> void thrust_div(float* srcData,
                            size_t size,
                            int value);
template <> void thrust_div(double* srcData,
                            size_t size,
                            int value);


template <typename T>
void thrust_combinedVar(T* /*var*/, T* /*mean*/, T* /*copyMean*/, size_t /*size*/) {}

template <> void thrust_combinedVar(half_float::half* var,
                                    half_float::half* mean,
                                    half_float::half* copyMean,
                                    size_t size);
template <> void thrust_combinedVar(float* var,
                                    float* mean,
                                    float* copyMean,
                                    size_t size);
template <> void thrust_combinedVar(double* var,
                                    double* mean,
                                    double* copyMean,
                                    size_t size);


}


#endif // N2D2_BATCHNORMCELL_FRAME_CUDA_KERNELS_H