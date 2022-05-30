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
