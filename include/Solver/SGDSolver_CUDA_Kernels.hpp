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

#ifndef N2D2_SGDSOLVER_CUDA_KERNELS_H
#define N2D2_SGDSOLVER_CUDA_KERNELS_H

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include "third_party/half.hpp"

namespace N2D2 {
template <class T>
void cudaClamp(T* x, unsigned int size, T minVal, T maxVal);
template <class T>
std::pair<T, T> cudaMinMax(T* x, unsigned int size);
template <class T>
void cudaPow(unsigned int size,
               T power,
               const T *x,
               T *y);
template <class T>
void cudaAdd(unsigned int size,
               T value,
               const T *x,
               T *y);
template <class T>
void cudaMult(unsigned int size,
               const T *x1,
               const T *x2,
               T *y);
template <class T>
void cudaInv(unsigned int size,
               const T *x,
               T *y);
}

#endif // N2D2_SGDSOLVER_CUDA_KERNELS_H
