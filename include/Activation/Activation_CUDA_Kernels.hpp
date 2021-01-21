/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_ACTIVATION_CUDA_KERNELS_H
#define N2D2_ACTIVATION_CUDA_KERNELS_H

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "CudaUtils.hpp"
#include "third_party/half.hpp"

#include "Solver/SGDSolver_CUDA_Kernels.hpp"

namespace N2D2 {
// Rectifier
template <class T>
void cudaRectifier_propagate(T* x,
                             T* y,
                             unsigned int size,
                             T leakSlope,
                             T clipping);
template <class T>
void cudaRectifier_backPropagate(T* y,
                                 T* dx,
                                 T* dy,
                                 unsigned int size,
                                 T leakSlope,
                                 T clipping);

// Saturation
template <class T>
void cudaSaturation_propagate(T* x,
                              T* y,
                              unsigned int size,
                              T threshold);
template <class T>
void cudaSaturation_backPropagate(T* y,
                                  T* dx,
                                  T* dy,
                                  unsigned int size,
                                  T threshold);

// Softplus
template <class T>
void cudaSoftplus_propagate(T* x, T* y, unsigned int size);
template <class T>
void cudaSoftplus_backPropagate(T* y,
                                T* dx,
                                T* dy,
                                unsigned int size);

// Swish
template <class T>
void cudaSwish_propagate(T* x,
                         T* y,
                         T* sigmoid,
                         unsigned int size);
template <class T>
void cudaSwish_backPropagate(T* y,
                             T* dx,
                             T* dy,
                             T* sigmoid,
                             unsigned int size);
}

#endif // N2D2_ACTIVATION_CUDA_KERNELS_H
