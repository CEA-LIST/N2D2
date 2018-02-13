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

#include "Activation/Activation_CUDA_Kernels.hpp"

// LeakyRectifier
__global__ void cudaSRectifier_propagate_kernel(float* x,
                                                unsigned int size,
                                                float leakSlope,
                                                int shifting,
                                                float clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (shifting > 0)
            x[i] /= (1 << shifting);
        else if (shifting < 0)
            x[i] *= (1 << (-shifting));

        if (clipping > 0.0)
            x[i] = (x[i] > 0.0f) ? min(x[i], clipping) : leakSlope * x[i];
        else
            x[i] = (x[i] > 0.0f) ? x[i] : leakSlope * x[i];
    }
}

__global__ void cudaDRectifier_propagate_kernel(double* x,
                                                unsigned int size,
                                                double leakSlope,
                                                int shifting,
                                                double clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (shifting > 0)
            x[i] /= (1 << shifting);
        else if (shifting < 0)
            x[i] *= (1 << (-shifting));

        if (clipping > 0.0)
            x[i] = (x[i] > 0.0) ? min(x[i], clipping) : leakSlope * x[i];
        else
            x[i] = (x[i] > 0.0) ? x[i] : leakSlope * x[i];
    }
}

__global__ void cudaSRectifier_backPropagate_kernel(float* x,
                                                    float* dx,
                                                    unsigned int size,
                                                    float leakSlope,
                                                    int shifting,
                                                    float clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (shifting > 0)
            dx[i] /= (1 << shifting);
        else if (shifting < 0)
            dx[i] *= (1 << (-shifting));

        if (clipping > 0.0) {
            dx[i] *= (x[i] > clipping) ? 0.0f : (x[i] > 0.0f)
                                       ? 1.0f
                                       : leakSlope;
        }
        else
            dx[i] *= (x[i] > 0.0f) ? 1.0f : leakSlope;
    }
}

__global__ void cudaDRectifier_backPropagate_kernel(double* x,
                                                    double* dx,
                                                    unsigned int size,
                                                    double leakSlope,
                                                    int shifting,
                                                    double clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (shifting > 0)
            dx[i] /= (1 << shifting);
        else if (shifting < 0)
            dx[i] *= (1 << (-shifting));

        if (clipping > 0.0) {
            dx[i] *= (x[i] > clipping) ? 0.0 : (x[i] > 0.0)
                                       ? 1.0
                                       : leakSlope;
        }
        else
            dx[i] *= (x[i] > 0.0) ? 1.0 : leakSlope;
    }
}

// Saturation
__global__ void cudaSSaturation_propagate_kernel(float* x,
                                                 unsigned int size,
                                                 int shifting,
                                                 float threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (shifting > 0)
            x[i] /= (1 << shifting);
        else if (shifting < 0)
            x[i] *= (1 << (-shifting));

        x[i] = (x[i] < -threshold) ? -threshold
             : (x[i] > threshold) ? threshold
             : x[i];
    }
}

__global__ void cudaDSaturation_propagate_kernel(double* x,
                                                 unsigned int size,
                                                 int shifting,
                                                 double threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (shifting > 0)
            x[i] /= (1 << shifting);
        else if (shifting < 0)
            x[i] *= (1 << (-shifting));

        x[i] = (x[i] < -threshold) ? -threshold
             : (x[i] > threshold) ? threshold
             : x[i];
    }
}

__global__ void
cudaSSaturation_backPropagate_kernel(float* x,
                                     float* dx,
                                     unsigned int size,
                                     int shifting,
                                     float threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (shifting > 0)
            dx[i] /= (1 << shifting);
        else if (shifting < 0)
            dx[i] *= (1 << (-shifting));

        dx[i] *= (x[i] > -threshold && x[i] < threshold)
            ? 1.0f : 0.0f;
    }
}

__global__ void
cudaDSaturation_backPropagate_kernel(double* x,
                                     double* dx,
                                     unsigned int size,
                                     int shifting,
                                     double threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (shifting > 0)
            dx[i] /= (1 << shifting);
        else if (shifting < 0)
            dx[i] *= (1 << (-shifting));

        dx[i] *= (x[i] > -threshold && x[i] < threshold)
            ? 1.0 : 0.0;
    }
}

// Softplus
__global__ void cudaSSoftplus_propagate_kernel(float* x, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        x[i] = log(1.0f + exp(x[i]));
    }
}

__global__ void cudaDSoftplus_propagate_kernel(double* x, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        x[i] = log(1.0 + exp(x[i]));
    }
}

__global__ void
cudaSSoftplus_backPropagate_kernel(float* x, float* dx, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dx[i] *= (1.0f - exp(-x[i]));
    }
}

__global__ void
cudaDSoftplus_backPropagate_kernel(double* x, double* dx, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dx[i] *= (1.0 - exp(-x[i]));
    }
}

// Rectifier
void N2D2::cudaSRectifier_propagate(float* x,
                                    unsigned int size,
                                    float leakSlope,
                                    int shifting,
                                    float clipping)
{
    cudaSRectifier_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, size, leakSlope, shifting, clipping);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDRectifier_propagate(double* x,
                                    unsigned int size,
                                    double leakSlope,
                                    int shifting,
                                    double clipping)
{
    cudaDRectifier_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, size, leakSlope, shifting, clipping);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSRectifier_backPropagate(float* x,
                                        float* dx,
                                        unsigned int size,
                                        float leakSlope,
                                        int shifting,
                                        float clipping)
{
    cudaSRectifier_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size, leakSlope, shifting, clipping);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDRectifier_backPropagate(double* x,
                                        double* dx,
                                        unsigned int size,
                                        double leakSlope,
                                        int shifting,
                                        double clipping)
{
    cudaDRectifier_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size, leakSlope, shifting, clipping);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

// Saturation
void N2D2::cudaSSaturation_propagate(float* x,
                                     unsigned int size,
                                     int shifting,
                                     float threshold)
{
    cudaSSaturation_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, size, shifting, threshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDSaturation_propagate(double* x,
                                     unsigned int size,
                                     int shifting,
                                     double threshold)
{
    cudaDSaturation_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, size, shifting, threshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSSaturation_backPropagate(float* x,
                                         float* dx,
                                         unsigned int size,
                                         int shifting,
                                         float threshold)
{
    cudaSSaturation_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size, shifting, threshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void
N2D2::cudaDSaturation_backPropagate(double* x,
                                    double* dx,
                                    unsigned int size,
                                    int shifting,
                                    double threshold)
{
    cudaDSaturation_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size, shifting, threshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

// Softplus
void N2D2::cudaSSoftplus_propagate(float* x, unsigned int size)
{
    cudaSSoftplus_propagate_kernel<<<(size + 255) / 256, 256>>>(x, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDSoftplus_propagate(double* x, unsigned int size)
{
    cudaDSoftplus_propagate_kernel<<<(size + 255) / 256, 256>>>(x, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSSoftplus_backPropagate(float* x, float* dx, unsigned int size)
{
    cudaSSoftplus_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDSoftplus_backPropagate(double* x, double* dx, unsigned int size)
{
    cudaDSoftplus_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
