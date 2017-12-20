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

#include "CudaUtils.hpp"

namespace N2D2 {
// Rectifier
void cudaSRectifier_propagate(float* x,
                              unsigned int size,
                              float leakSlope,
                              int shifting,
                              float clipping);
void cudaDRectifier_propagate(double* x,
                              unsigned int size,
                              double leakSlope,
                              int shifting,
                              double clipping);
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
void cudaSSaturation_propagate(float* x,
                               unsigned int size,
                               int shifting,
                               float threshold);
void cudaDSaturation_propagate(double* x,
                               unsigned int size,
                               int shifting,
                               double threshold);
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
void cudaSSoftplus_propagate(float* x, unsigned int size);
void cudaDSoftplus_propagate(double* x, unsigned int size);
void cudaSSoftplus_backPropagate(float* x, float* dx, unsigned int size);
void cudaDSoftplus_backPropagate(double* x, double* dx, unsigned int size);
}

#endif // N2D2_ACTIVATION_CUDA_KERNELS_H
