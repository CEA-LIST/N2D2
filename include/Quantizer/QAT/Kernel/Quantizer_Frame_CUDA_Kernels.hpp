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

#ifndef N2D2_QUANTIZER_FRAME_CUDA_KERNELS_H
#define N2D2_QUANTIZER_FRAME_CUDA_KERNELS_H

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>


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

namespace Quantizer_Frame_CUDA_Kernels {

/**
 * @brief   CUDA Kernel to calculate the mean in @p data 
 *          (half_float version)
 * @details No Trust methods for HALF type, 
 *          need to allocate buffer for partialSum
 * 
 * @param[in]   data        Pointer to the data vector
 * @param[in]   partialSum  Pointer to the partial sum vector
 *                          required to sum half values
 * @param[in]   size        Number of elements in data
 * @returns                 The mean of @p data
 */
half_float::half cudaH_mean(half_float::half* data, 
                            half_float::half* partialSum, 
                            const unsigned int size);

/**
 * @brief   CUDA Kernel to calculate the mean in @p data 
 *          (float version)
 * 
 * @param[in]   data   Pointer to the data vector
 * @param[in]   size   Number of elements in data
 * @returns            The mean of @p data
 */
float cudaF_mean(float* data, const unsigned int size);

/**
 * @brief   CUDA Kernel to calculate the mean in @p data 
 *          (double version)
 * 
 * @param[in]   data   Pointer to the data vector
 * @param[in]   size   Number of elements in data
 * @returns            The mean of @p data
 */
double cudaD_mean(double* data, const unsigned int size);

/**
 * @brief   CUDA Kernel to calculate the variance in @p data 
 *          (half_float version)
 * @details No Trust methods for HALF type, 
 *          need to allocate buffer for partialSum
 * 
 * @param[in]   data        Pointer to the data vector
 * @param[in]   partialSum  Pointer to the partial sum vector
 *                          required to sum half values
 * @param[in]   mean        Mean of the data vector
 * @param[in]   size        Number of elements in data
 * @returns                 The variance of @p data
 */
half_float::half cudaH_variance(half_float::half* data, 
                                half_float::half* partialSum, 
                                half_float::half mean,
                                const unsigned int size);

/**
 * @brief   CUDA Kernel to calculate the variance in @p data 
 *          (float version)
 * 
 * @param[in]   data   Pointer to the data vector
 * @param[in]   mean   Mean of the data vector
 * @param[in]   size   Number of elements in data
 * @returns            The variance of @p data
 */
float cudaF_variance(float* data, 
                     float mean, 
                     const unsigned int size);

/**
 * @brief   CUDA Kernel to calculate the variance in @p data 
 *          (double version)
 * 
 * @param[in]   data   Pointer to the data vector
 * @param[in]   mean   Mean of the data vector
 * @param[in]   size   Number of elements in data
 * @returns            The variance of @p data
 */
double cudaD_variance(double* data, 
                      double mean, 
                      const unsigned int size);
                                                 
/**
 * @brief   CUDA Kernel to reduce all elements in @p data 
 *          and return the result (half_float version)
 * @details No Trust methods for HALF type, 
 *          need to allocate buffer for partialSum
 * 
 * @param[in]   data        Pointer to the data vector
 * @param[in]   partialSum  Pointer to the partial sum vector
 *                          required to reduce half values
 * @param[in]   size        Number of elements in data
 * @returns                 Result of the reduction
 */
half_float::half cudaH_accumulate(half_float::half* data, 
                                  half_float::half* partialSum,
                                  const unsigned int size);

/**
 * @brief   CUDA Kernel to reduce all elements in @p data 
 *          and return the result (float version)
 * 
 * @param[in]   data   Pointer to the data vector
 * @param[in]   size   Number of elements in data
 * @returns            Result of the reduction
 */
float cudaF_accumulate(float* data, const unsigned int size);

/**
 * @brief   CUDA Kernel to reduce all elements in @p data 
 *          and return the result (double version)
 * 
 * @param[in]   data   Pointer to the data vector
 * @param[in]   size   Number of elements in data
 * @returns            Result of the reduction
 */
double cudaD_accumulate(double* data, const unsigned int size);

/**
 * @brief   CUDA Kernel to copy all elements in @p input to @p output
 *          (half_float version)
 * 
 * @param[in]   input       Pointer to the input vector
 * @param[out]  output      Pointer to the output vector
 * @param[in]   inputSize   Number of elements in input
 * @returns                 None
 */
void cudaH_copyData(half_float::half* input, 
                    half_float::half* output, 
                    unsigned int inputSize);

/**
 * @brief   CUDA Kernel to copy all elements in @p input to @p output
 *          (float version)
 * 
 * @param[in]   input       Pointer to the input vector
 * @param[out]  output      Pointer to the output vector
 * @param[in]   inputSize   Number of elements in input
 * @returns                 None
 */
void cudaF_copyData(float* input, 
                    float* output, 
                    unsigned int inputSize);

/**
 * @brief   CUDA Kernel to copy all elements in @p input to @p output
 *          (double version)
 * 
 * @param[in]   input       Pointer to the input vector
 * @param[out]  output      Pointer to the output vector
 * @param[in]   inputSize   Number of elements in input
 * @returns                 None
 */
void cudaD_copyData(double* input, 
                    double* output, 
                    unsigned int inputSize);

/**
 * @brief   CUDA Kernel to find the smallest and largest elements in @p data
 *          (half_float version)
 * 
 * @param[in]   data    Pointer to the data vector
 * @param[in]   size    Number of elements in data
 * @returns             A pair of values (minVal, maxVal) where minVal is the 
 *                      smallest element and maxVal is the largest element
 */
std::pair<half_float::half, half_float::half> cudaH_MinMax(half_float::half* data,
                                                           unsigned int size);

/**
 * @brief   CUDA Kernel to find the smallest and largest elements in @p data
 *          (float version)
 * 
 * @param[in]   data    Pointer to the data vector
 * @param[in]   size    Number of elements in data
 * @returns             A pair of values (minVal, maxVal) where minVal is the 
 *                      smallest element and maxVal is the largest element
 */
std::pair<float, float> cudaF_MinMax(float* data,
                                     unsigned int size);

/**
 * @brief   CUDA Kernel to find the smallest and largest elements in @p data
 *          (double version)
 * 
 * @param[in]   data    Pointer to the data vector
 * @param[in]   size    Number of elements in data
 * @returns             A pair of values (minVal, maxVal) where minVal is the 
 *                      smallest element and maxVal is the largest element
 */
std::pair<double, double> cudaD_MinMax(double* data,
                                       unsigned int size);

/**
 * @brief   CUDA Kernel to divide all elements in @p input by @p value
 *          (half_float version)
 * 
 * @param[in]   input   Pointer to the input vector
 * @param[in]   size    Number of elements in input
 * @param[in]   value   Constant used to divide
 * @returns             None
 */
void cudaH_div(half_float::half* data, 
               unsigned int size, 
               half_float::half value);

/**
 * @brief   CUDA Kernel to divide all elements in @p input by @p value
 *          (float version)
 * 
 * @param[in]   input   Pointer to the input vector
 * @param[in]   size    Number of elements in input
 * @param[in]   value   Constant used to divide
 * @returns             None
 */
void cudaF_div(float* data, 
               unsigned int size, 
               float value);

/**
 * @brief   CUDA Kernel to divide all elements in @p input by @p value
 *          (double version)
 * 
 * @param[in]   input   Pointer to the input vector
 * @param[in]   size    Number of elements in input
 * @param[in]   value   Constant used to divide
 * @returns             None
 */
void cudaD_div(double* data, 
               unsigned int size, 
               double value);

/**
 * @brief   CUDA Kernel to apply a tanh transformation to all elements 
 *          in @p input and store them to @p output
 *          (half_float version)
 * 
 * @param[in]   input   Pointer to the input vector
 * @param[out]  output  Pointer to the output vector
 * @param[in]   size    Number of elements in input
 * @returns             None
 */
void cudaH_tanh(half_float::half* input,
                half_float::half* output,
                unsigned int size);

/**
 * @brief   CUDA Kernel to apply a tanh transformation to all elements 
 *          in @p input and store them to @p output
 *          (float version)
 * 
 * @param[in]   input   Pointer to the input vector
 * @param[out]  output  Pointer to the output vector
 * @param[in]   size    Number of elements in input
 * @returns             None
 */
void cudaF_tanh(float* input,
                float* output,
                unsigned int size);

/**
 * @brief   CUDA Kernel to apply a tanh transformation to all elements 
 *          in @p input and store them to @p output
 *          (double version)
 * 
 * @param[in]   input   Pointer to the input vector
 * @param[out]  output  Pointer to the output vector
 * @param[in]   size    Number of elements in input
 * @returns             None
 */
void cudaD_tanh(double* input,
                double* output,
                unsigned int size);

}

}

#endif  // N2D2_QUANTIZER_FRAME_CUDA_KERNELS_H