/**
 * (C) Copyright 2020 CEA LIST. All Rights Reserved.
 *  Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
 *                  David BRIAND (david.briand@cea.fr)
 *                  Inna KUCHER (inna.kucher@cea.fr)
 *                  Olivier BICHLER (olivier.bichler@cea.fr)
 *                  Vincent TEMPLIER (vincent.templier@cea.fr)
 * 
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 * 
 */

#include "Quantizer/QAT/Kernel/SATQuantizer_Frame_CUDA_Kernels.hpp"
#include "Quantizer/QAT/Kernel/Quantizer_Frame_CUDA_Kernels.hpp"
#include "CudaUtils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace N2D2::Quantizer_Frame_CUDA_Kernels;

__global__ void cudaH_DorefaQ_kernel(__half* data,
                                     __half factor,
                                     float range,
                                     unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
    for (unsigned int i = index; i < size; i += stride) {
        __half q = __hmul(__float2half(0.5f), __hadd(__hdiv(data[i], factor),__float2half(1.0f)));

        // Check if q is between 0 and 1
        assert(__hge(q, __float2half(0.0f)) &&  __hle(q, __float2half(1.0f)));

        q = __hmul(__float2half(1.0f/range),hrint(__hmul(q,__float2half(range))));

        // Check if q is between 0 and 1
        assert(__hge(q, __float2half(0.0f)) &&  __hle(q, __float2half(1.0f)));

        data[i] =__hsub(__hmul(q,__float2half(2.0f)),__float2half(1.0f));
    }
#else
    for (unsigned int i = index; i < size; i += stride) {
        float q = 0.5f * (__half2float(data[i])/__half2float(factor) + 1.0f);

        // Check if q is between 0 and 1
        assert(q >= 0.0f && q <= 1.0f);

        q = (1.0f / range) * rintf(q * range);

        // Check if q is between 0 and 1
        assert(q >= 0.0f && q <= 1.0f);

        data[i] = __float2half(q * 2.0f - 1.0f);
    }
#endif
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaH_quantize_weight_default_propagate(half_float::half* weights,
                                                                                    half_float::half* weightsQ,
                                                                                    float range,
                                                                                    half_float::half* tanh_max_value,
                                                                                    unsigned int size)
{
    cudaH_tanh(weights, weightsQ, size);

    std::pair<half_float::half, half_float::half> tanh_minmax = cudaH_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaH_DorefaQ_kernel<<< (size + 255) / 256, 256>>>(reinterpret_cast<__half*>(weightsQ),
                                                       reinterpret_cast<__half&>(*tanh_max_value),
                                                       range,
                                                       size);

    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


__global__ void cudaF_DorefaQ_kernel(float* data,
                                     float factor,
                                     float range,
                                     unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        float q = 0.5f * ((data[i] / factor) + 1.0f);

        // Check if q is between 0 and 1
        assert(q >= 0.0f && q <= 1.0f);

        q = (1.0f / range) * rintf(q * range);

        // Check if q is between 0 and 1
        assert(q >= 0.0f && q <= 1.0f);

        data[i] = q * 2.0f - 1.0f;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_weight_default_propagate(float* weights,
                                                                                    float* weightsQ,
                                                                                    float range,
                                                                                    float* tanh_max_value,
                                                                                    unsigned int size)
{
    cudaF_tanh(weights, weightsQ, size);

    std::pair<float, float> tanh_minmax = cudaF_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaF_DorefaQ_kernel<<< (size + 255) / 256, 256>>>(weightsQ,
                                                       *tanh_max_value,
                                                       range,
                                                       size);

    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


__global__ void cudaD_DorefaQ_kernel(double* data,
                                     double factor,
                                     float range,
                                     unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        double q = 0.5 * ((data[i] / factor) + 1.0);

        // Check if q is between 0 and 1
        assert(q >= 0.0 && q <= 1.0);

        q = (1.0 / range) * llrint(q * range);

        // Check if q is between 0 and 1
        assert(q >= 0.0 && q <= 1.0);

        data[i] = q * 2.0 - 1.0;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaD_quantize_weight_default_propagate(double* weights,
                                                                                    double* weightsQ,
                                                                                    float range,
                                                                                    double* tanh_max_value,
                                                                                    unsigned int size)
{
    cudaD_tanh(weights, weightsQ, size);

    std::pair<double, double> tanh_minmax = cudaD_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaD_DorefaQ_kernel<<< (size + 255) / 256, 256>>>(weightsQ,
                                                       *tanh_max_value,
                                                       range,
                                                       size);

    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaH_weight_default_propagate(half_float::half* weights,
                                                                           half_float::half* weightsQ,
                                                                           float /*range*/,
                                                                           half_float::half* tanh_max_value,
                                                                           unsigned int size)
{
    cudaH_tanh(weights, weightsQ, size);

    std::pair<half_float::half, half_float::half> tanh_minmax = cudaH_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaH_div(weightsQ, size, *tanh_max_value);
}


void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_weight_default_propagate(float* weights,
                                                                           float* weightsQ,
                                                                           float /*range*/,
                                                                           float* tanh_max_value,
                                                                           unsigned int size)
{
    cudaF_tanh(weights, weightsQ, size);

    std::pair<float, float> tanh_minmax = cudaF_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaF_div(weightsQ, size, *tanh_max_value);
}


void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaD_weight_default_propagate(double* weights,
                                                                           double* weightsQ,
                                                                           float /*range*/,
                                                                           double* tanh_max_value,
                                                                           unsigned int size)
{
    cudaD_tanh(weights, weightsQ, size);

    std::pair<double, double> tanh_minmax = cudaD_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaD_div(weightsQ, size, *tanh_max_value);
}


__global__ void cudaF_DorefaQ_fullrange_kernel(float* data,
                                               float factor,
                                               float range,
                                               unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        float q = 0.5f * ((data[i] / factor) + 1.0f);

        // Check if q is between 0 and 1
        assert(q >= 0.0f && q <= 1.0f);

        q = (1.0f + 0.9998f / range) * q - (0.4999f / range);
        q = (1.0f / range) * rintf(q * range);

        // Check if q is between 0 and 1
        assert(q >= 0.0f && q <= 1.0f);

        data[i] = q * 2.0f - 1.0f;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_weight_fullrange_propagate(float* weights,
                                                                                      float* weightsQ,
                                                                                      float range,
                                                                                      float* tanh_max_value,
                                                                                      unsigned int size)
{
    cudaF_tanh(weights, weightsQ, size);

    std::pair<float, float> tanh_minmax = cudaF_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaF_DorefaQ_fullrange_kernel<<< (size + 255) / 256, 256>>>(weightsQ,
                                                                 *tanh_max_value,
                                                                 range,
                                                                 size);

    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


__global__ void cudaF_DorefaQ_symrange_kernel(float* data,
                                              float factor,
                                              float range,
                                              unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    range = floor(range / 2);

    for (unsigned int i = index; i < size; i += stride) {
        float q = data[i] / factor;

        // Check if q is between -1 and 1
        assert(q >= -1.0f && q <= 1.0f);

        q = (1.0f / range) * rintf(q * range);

        // Check if q is between -1 and 1
        assert(q >= -1.0f && q <= 1.0f);

        data[i] = q;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_weight_symrange_propagate(float* weights,
                                                                                     float* weightsQ,
                                                                                     float range,
                                                                                     float* tanh_max_value,
                                                                                     unsigned int size)
{
    cudaF_tanh(weights, weightsQ, size);

    std::pair<float, float> tanh_minmax = cudaF_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaF_DorefaQ_symrange_kernel<<< (size + 255) / 256, 256>>>(weightsQ,
                                                                *tanh_max_value,
                                                                range,
                                                                size);

    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


__global__ void cudaF_DorefaQ_asymrange_kernel(float* data,
                                               float factor,
                                               float range,
                                               unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    range = floor(range / 2);

    for (unsigned int i = index; i < size; i += stride) {
        float q = data[i] / factor;

        // Check if q is between -1 and 1
        assert(q >= -1.0f && q <= 1.0f);

        q = (1.0 + 1.0/(2.0 * range)) * q - (1.0/(2.0 * range));
        q = (1.0f / range) * rintf(q * range);

        // Check if q is between (-1 - 1/range) and 1
        assert(q >= -1.0f-(1.0f/range) && q <= 1.0f);

        data[i] = q;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_weight_asymrange_propagate(float* weights,
                                                                                      float* weightsQ,
                                                                                      float range,
                                                                                      float* tanh_max_value,
                                                                                      unsigned int size)
{
    cudaF_tanh(weights, weightsQ, size);

    std::pair<float, float> tanh_minmax = cudaF_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaF_DorefaQ_asymrange_kernel<<< (size + 255) / 256, 256>>>(weightsQ,
                                                                 *tanh_max_value,
                                                                 range,
                                                                 size);

    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


__global__ void cudaF_asymrange_kernel(float* data,
                                       float factor,
                                       float range,
                                       unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    range = floor(range / 2);

    for (unsigned int i = index; i < size; i += stride) {
        float q = data[i] / factor;

        // Check if q is between -1 and 1
        assert(q >= -1.0f && q <= 1.0f);

        q = (1.0 + 1.0/(2.0 * range)) * q - (1.0/(2.0 * range));

        data[i] = q;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_weight_asymrange_propagate(float* weights,
                                                                             float* weightsQ,
                                                                             float range,
                                                                             float* tanh_max_value,
                                                                             unsigned int size)
{
    cudaF_tanh(weights, weightsQ, size);

    std::pair<float, float> tanh_minmax = cudaF_MinMax(weightsQ, size);
    *tanh_max_value = max(abs(tanh_minmax.first), abs(tanh_minmax.second));

    cudaF_asymrange_kernel<<< (size + 255) / 256, 256>>>(weightsQ,
                                                         *tanh_max_value,
                                                         range,
                                                         size);

    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


// ----------------------------------------------------------------------------
// ---------------------------- SAT SCALING KERNEL ----------------------------
// ----------------------------------------------------------------------------

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaH_apply_scaling(half_float::half* data,
                                                                half_float::half* scaling_value,
                                                                half_float::half* partial_sum,
                                                                unsigned int scaling_factor,
                                                                unsigned int size)
{
    half_float::half mean = Quantizer_Frame_CUDA_Kernels::cudaH_mean(data, partial_sum, size);
    half_float::half variance = Quantizer_Frame_CUDA_Kernels::cudaH_variance(data, partial_sum, mean, size);

    *scaling_value = sqrt(variance * scaling_factor);

    cudaH_div(data, size, *scaling_value);
}


void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_apply_scaling(float* data,
                                                                float* scaling_value,
                                                                unsigned int scaling_factor,
                                                                unsigned int size)
{
    float mean = Quantizer_Frame_CUDA_Kernels::cudaF_mean(data, size);
    float variance = Quantizer_Frame_CUDA_Kernels::cudaF_variance(data, mean, size);

    *scaling_value = sqrtf(variance * scaling_factor);

    cudaF_div(data, size, *scaling_value);
}


void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaD_apply_scaling(double* data,
                                                                double* scaling_value,
                                                                unsigned int scaling_factor,
                                                                unsigned int size)
{
    double mean = Quantizer_Frame_CUDA_Kernels::cudaD_mean(data, size);
    double variance = Quantizer_Frame_CUDA_Kernels::cudaD_variance(data, mean, size);

    *scaling_value = sqrt(variance * scaling_factor);

    cudaD_div(data, size, *scaling_value);
}


// ----------------------------------------------------------------------------
// --------------------- SAT ACTIVATION FORWARD KERNEL ------------------------
// -------------------------- HALF FLOAT VERSION ------------------------------
// ----------------------------------------------------------------------------


__global__ void cudaH_CGPACT_kernel(__half* data,
                                    const float range,
                                    const __half* alpha,
                                    unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
    const __half alpha_ = __float2half(abs(__half2float(alpha[0])));

    for (unsigned int i = index; i < size; i += stride) {

        const __half x = data[i];
        const __half x_clip = (__hlt(x, __float2half(0.0f))) ? __float2half(0.0f) : __hlt(x, alpha_) ? x : alpha_;
        __half q = __hdiv(x_clip, alpha_);

        // Test if q is in [0;1] before rounding
        assert(__hge(q, __float2half(0.0f)) &&  __hle(q, __float2half(1.0f)));

        q = __hmul(__float2half(1.0f / range),hrint(__hmul(q,__float2half(range))));

        // Test if q is in [0;1] after rounding
        assert(__hge(q, __float2half(0.0f)) &&  __hle(q, __float2half(1.0f)));

        data[i] = __hmul(q, alpha_);
    }
#else
    const float alpha_ = abs(__half2float(alpha[0]));

    for (unsigned int i = index; i < size; i += stride) {
        
        float x = __half2float(data[i]);
        float x_clip = (x < 0.0f) ? 0.0f : (x < alpha_) ? x : alpha_;
        float q = x_clip / alpha_;

        // Test if q is in [0;1] before rounding
        assert(q >= 0.0f && q <= 1.0f);

        q = (1.0f / range) * rint(q * range);

        // Test if q is in [0;1] after rounding
        assert(q >= 0.0f && q <= 1.0f);

        data[i] = __float2half(q * alpha_);
    }
#endif
}


__global__ void cudaH_CGPACT_save_kernel(__half* data,
                                         const float range,
                                         const __half* alpha,
                                         __half* fpData,
                                         unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
    const __half alpha_ = __float2half(abs(__half2float(alpha[0])));

    for (unsigned int i = index; i < size; i += stride) {

        // Save full precision data value before quantization
        fpData[i] = data[i];

        const __half x = data[i];
        const __half x_clip = (__hlt(x, __float2half(0.0f))) ? __float2half(0.0f) : __hlt(x, alpha_) ? x : alpha_;
        __half q = x_clip / alpha_;

        // Test if q is in [0;1] before rounding
        assert(__hge(q, __float2half(0.0f)) &&  __hle(q, __float2half(1.0f)));

        q = __hmul(__float2half(1.0f / range),hrint(__hmul(q,__float2half(range))));

        // Test if q is in [0;1] after rounding
        assert(__hge(q, __float2half(0.0f)) &&  __hle(q, __float2half(1.0f)));

        data[i] = __hmul(q, alpha_);
    }
#else
    const float alpha_ = abs(__half2float(alpha[0]));

    for (unsigned int i = index; i < size; i += stride) {

        // Save full precision data value before quantization
        fpData[i] = data[i];
        
        float x = __half2float(data[i]);
        float x_clip = (x < 0.0f) ? 0.0f : (x < alpha_) ? x : alpha_;
        float q = x_clip / alpha_;

        // Test if q is in [0;1] before rounding
        assert(q >= 0.0f && q <= 1.0f);

        q = (1.0f / range) * rint(q * range);

        // Test if q is in [0;1] after rounding
        assert(q >= 0.0f && q <= 1.0f);

        data[i] = __float2half(q * alpha_);
    }
#endif
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaH_quantize_activation_propagate(half_float::half* activations,
                                                                                float range,
                                                                                const half_float::half* alpha,
                                                                                half_float::half* fpActivations,
                                                                                unsigned int size,
                                                                                bool inference)
{
    if (inference) {
        cudaH_CGPACT_kernel<<< (size + 255) / 256, 256>>> (reinterpret_cast<__half*> (activations), 
                                                           range, 
                                                           reinterpret_cast<const __half*> (alpha), 
                                                           size);
    } else {
        cudaH_CGPACT_save_kernel<<< (size + 255) / 256, 256>>> (reinterpret_cast<__half*> (activations),
                                                                range,
                                                                reinterpret_cast<const __half*> (alpha),
                                                                reinterpret_cast<__half*> (fpActivations),
                                                                size);
    }
    
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}



// ----------------------------------------------------------------------------
// --------------------- SAT ACTIVATION FORWARD KERNEL ------------------------
// ----------------------------- FLOAT VERSION --------------------------------
// ----------------------------------------------------------------------------


__global__ void cudaF_CGPACT_kernel(float* data,
                                    const float range,
                                    const float* alpha,
                                    unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const float alpha_ = abs(alpha[0]);

    for (unsigned int i = index; i < size; i += stride) {
        
        const float x = data[i];
        const float x_clip = (x < 0.0f) ? 0.0f : (x < alpha_) ? x : alpha_;
        float q = x_clip / alpha_;

        // Test if q is in [0;1] before rounding
        assert(q >= 0.0f && q <= 1.0f);
        
        q = rintf(range * q) / range;

        // Test if q is in [0;1] after rounding
        assert(q >= 0.0f && q <= 1.0f);

        data[i] = q * alpha_;
    }
}

__global__ void cudaF_CGPACT_save_kernel(float* data,
                                         const float range,
                                         const float* alpha,
                                         float* fpData,
                                         unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const float alpha_ = abs(alpha[0]);

    for (unsigned int i = index; i < size; i += stride) {

        // Save full precision data value before quantization
        fpData[i] = data[i];
        
        const float x = data[i];
        const float x_clip = (x < 0.0f) ? 0.0f : (x < alpha_) ? x : alpha_;
        float q = x_clip / alpha_;

        // Test if q is in [0;1] before rounding
        assert(q >= 0.0f && q <= 1.0f);
        
        q = rintf(range * q) / range;

        // Test if q is in [0;1] after rounding
        assert(q >= 0.0f && q <= 1.0f);

        data[i] = q * alpha_;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_activation_propagate(float* activations,
                                                                                float range,
                                                                                const float* alpha,
                                                                                float* fpActivations,
                                                                                unsigned int size,
                                                                                bool inference)
{
    if (inference) {
        cudaF_CGPACT_kernel<<< (size + 255) / 256, 256>>> (activations, range, alpha, size);
    } else {
        cudaF_CGPACT_save_kernel<<< (size + 255) / 256, 256>>> (activations,
                                                                range,
                                                                alpha,
                                                                fpActivations,
                                                                size);
    }
    
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}



// ----------------------------------------------------------------------------
// --------------------- SAT ACTIVATION FORWARD KERNEL ------------------------
// ---------------------------- DOUBLE VERSION --------------------------------
// ----------------------------------------------------------------------------


__global__ void cudaD_CGPACT_kernel(double* data,
                                    const float range,
                                    const double* alpha,
                                    unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const double alpha_ = abs(alpha[0]);

    for (unsigned int i = index; i < size; i += stride) {
        
        const double x = data[i];
        const double x_clip = (x < 0.0) ? 0.0 : (x < alpha_) ? x : alpha_;
        double q = x_clip / alpha_;

        // Test if q is in [0;1] before rounding
        assert(q >= 0.0 && q <= 1.0);
        
        q = llrint(range * q) / range;

        // Test if q is in [0;1] after rounding
        assert(q >= 0.0 && q <= 1.0);

        data[i] = q * alpha_;
    }
}

__global__ void cudaD_CGPACT_save_kernel(double* data,
                                         const float range,
                                         const double* alpha,
                                         double* fpData,
                                         unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const double alpha_ = abs(alpha[0]);

    for (unsigned int i = index; i < size; i += stride) {

        // Save full precision data value before quantization
        fpData[i] = data[i];
        
        const double x = data[i];
        const double x_clip = (x < 0.0) ? 0.0 : (x < alpha_) ? x : alpha_;
        double q = x_clip / alpha_;

        // Test if q is in [0;1] before rounding
        assert(q >= 0.0 && q <= 1.0);
        
        q = llrint(range * q) / range;

        // Test if q is in [0;1] after rounding
        assert(q >= 0.0 && q <= 1.0);

        data[i] = q * alpha_;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaD_quantize_activation_propagate(double* activations,
                                                                                float range,
                                                                                const double* alpha,
                                                                                double* fpActivations,
                                                                                unsigned int size,
                                                                                bool inference)
{
    if (inference) {
        cudaD_CGPACT_kernel<<< (size + 255) / 256, 256>>> (activations, range, alpha, size);
    } else {
        cudaD_CGPACT_save_kernel<<< (size + 255) / 256, 256>>> (activations,
                                                                range,
                                                                alpha,
                                                                fpActivations,
                                                                size);
    }
    
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


// ----------------------------------------------------------------------------
// ---------------- WEIGHT DEFAULT/SYMRANGE BACKWARD KERNEL -------------------
// -------------------------- HALF FLOAT VERSION ------------------------------
// ----------------------------------------------------------------------------

__global__ void cudaH_SATGrad_kernel(__half* diffInputs,
                                     __half* diffOutputs,
                                     __half* x,
                                     __half factor,
                                     unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
    for (unsigned int i = index; i < size; i += stride) {
        __half inv_cosh = __float2half(1/std::cosh(__half2float(x[i])));
        __half grad = __hdiv(__hmul(inv_cosh, inv_cosh), factor);
        diffOutputs[i] = __hmul(diffInputs[i], grad);
    }
#else
    for (unsigned int i = index; i < size; i += stride) {
        float inv_cosh = 1/std::cosh(__half2float(x[i]));
        float grad = inv_cosh * inv_cosh * (1/__half2float(factor));
        diffOutputs[i] = __float2half(__half2float(diffInputs[i]) * grad);
    }
#endif
}                                     

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaH_quantize_weight_default_back_propagate(half_float::half* diffInputs,
                                                                                         half_float::half* diffOutputs,
                                                                                         half_float::half* fpWeights,
                                                                                         half_float::half factor,
                                                                                         unsigned int size)
{
    cudaH_SATGrad_kernel<<< (size + 255) / 256, 256>>> (reinterpret_cast<__half*>(diffInputs),
                                                        reinterpret_cast<__half*>(diffOutputs),
                                                        reinterpret_cast<__half*>(fpWeights),
                                                        reinterpret_cast<__half&>(factor),
                                                        size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


// ----------------------------------------------------------------------------
// ---------------- WEIGHT DEFAULT/SYMRANGE BACKWARD KERNEL -------------------
// ----------------------------- FLOAT VERSION --------------------------------
// ----------------------------------------------------------------------------

__global__ void cudaF_SATGrad_kernel(float* diffInputs,
                                     float* diffOutputs,
                                     float* x,
                                     float factor,
                                     unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        float inv_cosh = 1/std::cosh(x[i]);
        float grad = inv_cosh * inv_cosh * (1/factor);
        diffOutputs[i] = diffInputs[i] * grad;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_weight_default_back_propagate(float* diffInputs,
                                                                                         float* diffOutputs,
                                                                                         float* fpWeights,
                                                                                         float factor,
                                                                                         unsigned int size)
{
    cudaF_SATGrad_kernel<<< (size + 255) / 256, 256>>> (diffInputs, diffOutputs, fpWeights, factor, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


// ----------------------------------------------------------------------------
// ---------------- WEIGHT DEFAULT/SYMRANGE BACKWARD KERNEL -------------------
// ----------------------------- DOUBLE VERSION -------------------------------
// ----------------------------------------------------------------------------

__global__ void cudaD_SATGrad_kernel(double* diffInputs,
                                     double* diffOutputs,
                                     double* x,
                                     double factor,
                                     unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        double inv_cosh = 1/std::cosh(x[i]);
        double grad = inv_cosh * inv_cosh * (1/factor);
        diffOutputs[i] = diffInputs[i] * grad;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaD_quantize_weight_default_back_propagate(double* diffInputs,
                                                                                         double* diffOutputs,
                                                                                         double* fpWeights,
                                                                                         double factor,
                                                                                         unsigned int size)
{
    cudaD_SATGrad_kernel<<< (size + 255) / 256, 256>>> (diffInputs, diffOutputs, fpWeights, factor, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


// ----------------------------------------------------------------------------
// -------------------- WEIGHT FULLRANGE BACKWARD KERNEL ----------------------
// ----------------------------- FLOAT VERSION --------------------------------
// ----------------------------------------------------------------------------

__global__ void cudaF_SATGrad_FullRange_kernel(float* diffInputs,
                                               float* diffOutputs,
                                               float* x,
                                               float range,
                                               float factor,
                                               unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    float fullrange_factor = (1.0f + 0.9998f/range);

    for (unsigned int i = index; i < size; i += stride) {
        float inv_cosh = 1/std::cosh(x[i]);
        float grad = inv_cosh * inv_cosh * (1/factor) * fullrange_factor;
        diffOutputs[i] = diffInputs[i] * grad;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_weight_fullrange_back_propagate(float* diffInputs,
                                                                                           float* diffOutputs,
                                                                                           float* fpWeights,
                                                                                           float range,
                                                                                           float factor,
                                                                                           unsigned int size)
{
    cudaF_SATGrad_FullRange_kernel<<< (size + 255) / 256, 256>>> (diffInputs, diffOutputs, fpWeights, 
                                                                  range, factor, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}  


// ----------------------------------------------------------------------------
// -------------------- WEIGHT ASYMRANGE BACKWARD KERNEL ----------------------
// ----------------------------- FLOAT VERSION --------------------------------
// ----------------------------------------------------------------------------

__global__ void cudaF_SATGrad_AsymRange_kernel(float* diffInputs,
                                               float* diffOutputs,
                                               float* x,
                                               float range,
                                               float factor,
                                               unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    range = floor(range / 2);
    float asymm_factor = 1.0f + 1.0f/(2.0f * range);

    for (unsigned int i = index; i < size; i += stride) {
        float inv_cosh = 1/std::cosh(x[i]);
        float grad = inv_cosh * inv_cosh * (1/factor) * asymm_factor;
        diffOutputs[i] = diffInputs[i] * grad;
    }
}

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_weight_asymrange_back_propagate(float* diffInputs,
                                                                                           float* diffOutputs,
                                                                                           float* fpWeights,
                                                                                           float range,
                                                                                           float factor,
                                                                                           unsigned int size)
{
    cudaF_SATGrad_AsymRange_kernel<<< (size + 255) / 256, 256>>> (diffInputs, diffOutputs, fpWeights, 
                                                                  range, factor, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}  


// ----------------------------------------------------------------------------
// --------------------- SAT ACTIVATION BACKWARD KERNEL -----------------------
// -------------------------- HALF FLOAT VERSION ------------------------------
// ----------------------------------------------------------------------------


__global__ void cudaH_AlphaGrad_kernel(const __half* diffInput,
                                       __half* diffAlpha,
                                       const __half* fpActivations,
                                       const float range,
                                       const __half* alpha,
                                       unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
    const __half alpha_ = __float2half(abs(__half2float(alpha[0])));

    for (unsigned int i = index; i < size; i += stride) {

        const __half x = fpActivations[i];
        const __half x_clip = __hlt(x, __float2half(0.0f)) ? __float2half(0.0f) : __hlt(x, alpha_) ? x : alpha_;
        __half q = __hdiv(x_clip, alpha_);

        // Test if q is in [0;1]
        assert(__hge(q, __float2half(0.0f)) &&  __hle(q, __float2half(1.0f)));

        __half qData = __hmul(__float2half(1.0f / range), hrint(__hmul(q, __float2half(range))));

        // Test if qData is in [0;1]
        assert(__hge(qData, __float2half(0.0f)) &&  __hle(qData, __float2half(1.0f)));

        const __half dQ = __hge(x, alpha_) ? __float2half(1.0f) : __hsub(qData, q);
        diffAlpha[i] = dQ * diffInput[i]; 
    }
#else
    const float alpha_ = abs(__half2float(alpha[0]));

    for (unsigned int i = index; i < size; i += stride) {

        const float x = __half2float(fpActivations[i]);
        const float x_clip = (x < 0.0f) ? 0.0f : (x < alpha_) ? x : alpha_;
        const float q = x_clip / alpha_;

        // Test if q is in [0;1]
        assert(q >= 0.0f && q <= 1.0f);
        
        float qData = rintf(q * range) / range;

        // Test if qData is in [0;1]
        assert(qData >= 0.0f && qData <= 1.0f);

        const float dQ = (x >= alpha_) ? 1.0f : (qData - q);
        diffAlpha[i] = __float2half(dQ * __half2float(diffInput[i])); 
    }
#endif
}


__global__ void cudaH_CGPACTGrad_kernel(__half* diffInput,
                                        __half* diffOutput,
                                        const __half* fpActivations,
                                        const __half* alpha,
                                        unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
    const __half alpha_ = __float2half(abs(__half2float(alpha[0])));

    for (unsigned int i = index; i < size; i += stride) {
        // STE
        const __half dQ = __hle(fpActivations[i],__float2half(0.0f)) 
                            ? __float2half(0.0f)
                            : __hgt(fpActivations[i], alpha_) 
                                ?  __float2half(0.0f) :  __float2half(1.0f);
        diffOutput[i] = __hmul(dQ, diffInput[i]); 
    } 
#else
    const float alpha_ = abs(__half2float(alpha[0]));

    for (unsigned int i = index; i < size; i += stride) {
        // STE
        const float dQ = __half2float(fpActivations[i]) <= 0.0f 
                        ? 0.0f 
                        : __half2float(fpActivations[i]) > alpha_ 
                            ? 0.0f : 1.0f;
        diffOutput[i] = __float2half(dQ * __half2float(diffInput[i]));
    }
#endif
}


void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaH_quantize_activation_back_propagate(half_float::half* diffInput,
                                                                                     half_float::half* diffOutput,
                                                                                     half_float::half* diffAlpha,
                                                                                     const half_float::half* fpActivations,
                                                                                     float range,
                                                                                     const half_float::half* alpha,
                                                                                     unsigned int size)
{
    cudaH_AlphaGrad_kernel<<< (size + 255) / 256, 256>>> (reinterpret_cast<__half*> (diffInput),
                                                          reinterpret_cast<__half*> (diffAlpha),
                                                          reinterpret_cast<const __half*> (fpActivations),
                                                          range,
                                                          reinterpret_cast<const __half*> (alpha),
                                                          size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

    cudaH_CGPACTGrad_kernel<<< (size + 255) / 256, 256>>> (reinterpret_cast<__half*> (diffInput),
                                                           reinterpret_cast<__half*> (diffOutput),
                                                           reinterpret_cast<const __half*> (fpActivations),
                                                           reinterpret_cast<const __half*> (alpha),
                                                           size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


// ----------------------------------------------------------------------------
// --------------------- SAT ACTIVATION BACKWARD KERNEL -----------------------
// ----------------------------- FLOAT VERSION --------------------------------
// ----------------------------------------------------------------------------


__global__ void cudaF_AlphaGrad_kernel(const float* diffInput,
                                       float* diffAlpha,
                                       const float* fpActivations,
                                       const float range,
                                       const float* alpha,
                                       unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const float alpha_ = abs(alpha[0]);

    for (unsigned int i = index; i < size; i += stride) {
        
        const float x = fpActivations[i];
        const float x_clip = (x < 0.0f) ? 0.0f : (x < alpha_) ? x : alpha_;
        const float q = x_clip / alpha_;

        assert(q >= 0.0f && q <= 1.0f);

        float qData = rintf(q * range) / range;

        assert(qData >= 0.0f && qData <= 1.0f);

        const float dQ = (x >= alpha_) ? 1.0f : (qData - q);
        diffAlpha[i] = dQ * diffInput[i];
    }
} 


__global__ void cudaF_CGPACTGrad_kernel(float* diffInput,
                                        float* diffOutput,
                                        const float* fpActivations,
                                        const float* alpha,
                                        unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const float alpha_ = abs(alpha[0]);

    for (unsigned int i = index; i < size; i += stride) {
        // STE
        const float dQ = fpActivations[i] <= 0.0f 
                        ? 0.0f 
                        : fpActivations[i] > alpha_ 
                            ? 0.0f : 1.0f;

        diffOutput[i] = dQ * diffInput[i]; 
    }
} 

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_activation_back_propagate(float* diffInput,
                                                                                     float* diffOutput,
                                                                                     float* diffAlpha,
                                                                                     const float* fpActivations,
                                                                                     float range,
                                                                                     const float* alpha,
                                                                                     unsigned int size)
{
    cudaF_AlphaGrad_kernel<<< (size + 255) / 256, 256>>> (diffInput,
                                                          diffAlpha,
                                                          fpActivations,
                                                          range,
                                                          alpha,
                                                          size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

    cudaF_CGPACTGrad_kernel<<< (size + 255) / 256, 256>>> (diffInput,
                                                           diffOutput,
                                                           fpActivations,
                                                           alpha,
                                                           size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}



// ----------------------------------------------------------------------------
// --------------------- SAT ACTIVATION BACKWARD KERNEL -----------------------
// ----------------------------- DOUBLE VERSION -------------------------------
// ----------------------------------------------------------------------------


__global__ void cudaD_AlphaGrad_kernel(const double* diffInput,
                                       double* diffAlpha,
                                       const double* fpActivations,
                                       const float range,
                                       const double* alpha,
                                       unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const double alpha_ = abs(alpha[0]);

    for (unsigned int i = index; i < size; i += stride) {
        
        const double x = fpActivations[i];
        const double x_clip = (x < 0.0) ? 0.0 : (x < alpha_) ? x : alpha_;
        const double q = x_clip / alpha_;

        assert(q >= 0.0 && q <= 1.0);

        double qData = llrint(q * range) / range;

        assert(qData >= 0.0 && qData <= 1.0);

        const double dQ = (x >= alpha_) ? 1.0 : (qData - q);
        diffAlpha[i] = dQ * diffInput[i];
    }
} 


__global__ void cudaD_CGPACTGrad_kernel(double* diffInput,
                                        double* diffOutput,
                                        const double* fpActivations,
                                        const double* alpha,
                                        unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const double alpha_ = abs(alpha[0]);

    for (unsigned int i = index; i < size; i += stride) {
        // STE
        const double dQ = fpActivations[i] <= 0.0
                        ? 0.0 
                        : fpActivations[i] > alpha_ 
                            ? 0.0 : 1.0;

        diffOutput[i] = dQ * diffInput[i]; 
    }
} 

void N2D2::SATQuantizer_Frame_CUDA_Kernels::cudaD_quantize_activation_back_propagate(double* diffInput,
                                                                                     double* diffOutput,
                                                                                     double* diffAlpha,
                                                                                     const double* fpActivations,
                                                                                     float range,
                                                                                     const double* alpha,
                                                                                     unsigned int size)
{
    cudaD_AlphaGrad_kernel<<< (size + 255) / 256, 256>>> (diffInput,
                                                          diffAlpha,
                                                          fpActivations,
                                                          range,
                                                          alpha,
                                                          size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

    cudaD_CGPACTGrad_kernel<<< (size + 255) / 256, 256>>> (diffInput,
                                                           diffOutput,
                                                           fpActivations,
                                                           alpha,
                                                           size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
