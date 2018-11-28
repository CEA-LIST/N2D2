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
void cudaHRectifier_propagate(half_float::half* x,
                              half_float::half* y,
                              unsigned int size,
                              half_float::half leakSlope,
                              int shifting,
                              half_float::half clipping);
void cudaSRectifier_propagate(float* x,
                              float* y,
                              unsigned int size,
                              float leakSlope,
                              int shifting,
                              float clipping);
void cudaDRectifier_propagate(double* x,
                              double* y,
                              unsigned int size,
                              double leakSlope,
                              int shifting,
                              double clipping);
void cudaHRectifier_backPropagate(half_float::half* x,
                                  half_float::half* dx,
                                  unsigned int size,
                                  half_float::half leakSlope,
                                  int shifting,
                                  half_float::half clipping);
void cudaSRectifier_backPropagate(float* x,
                                  float* dx,
                                  unsigned int size,
                                  float leakSlope,
                                  int shifting,
                                  float clipping);
void cudaDRectifier_backPropagate(double* x,
                                  double* dx,
                                  unsigned int size,
                                  double leakSlope,
                                  int shifting,
                                  double clipping);
// Saturation
void cudaHSaturation_propagate(half_float::half* x,
                               half_float::half* y,
                               unsigned int size,
                               int shifting,
                               half_float::half threshold);
void cudaSSaturation_propagate(float* x,
                               float* y,
                               unsigned int size,
                               int shifting,
                               float threshold);
void cudaDSaturation_propagate(double* x,
                               double* y,
                               unsigned int size,
                               int shifting,
                               double threshold);
void cudaHSaturation_backPropagate(half_float::half* x,
                                   half_float::half* dx,
                                   unsigned int size,
                                   int shifting,
                                   half_float::half threshold);
void cudaSSaturation_backPropagate(float* x,
                                   float* dx,
                                   unsigned int size,
                                   int shifting,
                                   float threshold);
void cudaDSaturation_backPropagate(double* x,
                                   double* dx,
                                   unsigned int size,
                                   int shifting,
                                   double threshold);
// Softplus
void cudaHSoftplus_propagate(half_float::half* x,
                             half_float::half* y,
                             unsigned int size);
void cudaSSoftplus_propagate(float* x, float* y, unsigned int size);
void cudaDSoftplus_propagate(double* x, double* y, unsigned int size);
void cudaHSoftplus_backPropagate(half_float::half* x, half_float::half* dx,
                                 unsigned int size);
void cudaSSoftplus_backPropagate(float* x, float* dx, unsigned int size);
void cudaDSoftplus_backPropagate(double* x, double* dx, unsigned int size);
}

#endif // N2D2_ACTIVATION_CUDA_KERNELS_H
