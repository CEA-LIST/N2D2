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
void cudaHclamp(half_float::half* x, unsigned int size,
                half_float::half minVal, half_float::half maxVal);
void cudaSclamp(float* x, unsigned int size, float minVal, float maxVal);
void cudaDclamp(double* x, unsigned int size, double minVal, double maxVal);
std::pair<half_float::half, half_float::half>
cudaHminMax(half_float::half* x, unsigned int size);
std::pair<float, float> cudaSminMax(float* x, unsigned int size);
std::pair<double, double> cudaDminMax(double* x, unsigned int size);
void cudaHquantize(half_float::half* x,
                   half_float::half* y,
                   unsigned int size,
                   half_float::half minVal,
                   half_float::half maxVal,
                   unsigned int quantizationLevels,
                   bool truncate = false);
void cudaSquantize(float* x,
                   float* y,
                   unsigned int size,
                   float minVal,
                   float maxVal,
                   unsigned int quantizationLevels,
                   bool truncate = false);
void cudaDquantize(double* x,
                   double* y,
                   unsigned int size,
                   double minVal,
                   double maxVal,
                   unsigned int quantizationLevels,
                   bool truncate = false);
void cudaHscal(unsigned int size,
               half_float::half alpha,
               half_float::half *x);
void cudaHaxpy(unsigned int size,
               half_float::half alpha,
               const half_float::half *x,
               half_float::half *y);
void cudaHpow(unsigned int size,
               half_float::half power,
               const half_float::half *x,
               half_float::half *y);
void cudaSpow(unsigned int size,
               float power,
               const float *x,
               float *y);
void cudaDpow(unsigned int size,
               double power,
               const double *x,
               double *y);
void cudaHadd(unsigned int size,
               half_float::half value,
               const half_float::half *x,
               half_float::half *y);
void cudaSadd(unsigned int size,
               float value,
               const float *x,
               float *y);
void cudaDadd(unsigned int size,
               double value,
               const double *x,
               double *y);
void cudaHmult(unsigned int size,
               const half_float::half *x1,
               const half_float::half *x2,
               half_float::half *y);
void cudaSmult(unsigned int size,
               const float *x1,
               const float *x2,
               float *y);
void cudaDmult(unsigned int size,
               const double *x1,
               const double *x2,
               double *y);
void cudaHinv(unsigned int size,
               const half_float::half *x,
               half_float::half *y);
void cudaSinv(unsigned int size,
               const float *x,
               float *y);
void cudaDinv(unsigned int size,
               const double *x,
               double *y);
}

#endif // N2D2_SGDSOLVER_CUDA_KERNELS_H
