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

#include "Quantizer/QAT/Kernel/LSQQuantizer_Frame_CUDA_Kernels.hpp"
#include "CudaUtils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


__global__ void cudaF_LSQ_propagate_kernel(const float* data_,
                                           const float* stepSize,
                                           const int rangeMin,
                                           const int rangeMax,
                                           float* qData_,
                                           float* fpData_,
                                           bool saveFpData_,
                                           const size_t size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < size; i += stride) {
        float qData = data_[i] / stepSize[0];
        qData = (qData <= (float) rangeMin ) ? (float) rangeMin :
                (qData >= (float) rangeMax ) ? (float) rangeMax :
                qData;
        qData = rintf(qData);
        if(saveFpData_) {
            fpData_[i] = data_[i];
        }
        qData_[i] = qData*stepSize[0];
    }
} 


void N2D2::LSQQuantizer_Frame_CUDA_Kernels::cudaF_quantize_propagate(const float* fullPrecData,
                                                                     const float* stepSize,
                                                                     const int rangeMin,
                                                                     const int rangeMax,
                                                                     float* quantData,
                                                                     float* fullPrecDataCopy,
                                                                     bool saveCopy,
                                                                     const size_t inputSize)
{
    cudaF_LSQ_propagate_kernel<<< (inputSize + 255) / 256, 256>>>(fullPrecData,
                                                                  stepSize,
                                                                  rangeMin,
                                                                  rangeMax,
                                                                  quantData,
                                                                  fullPrecDataCopy,
                                                                  saveCopy,
                                                                  inputSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


__global__ void cudaF_LSQ_back_propagate_kernel(float* diffQuantData_,
                                                const float* fullPrecData_,
                                                float* diffFullPrecData_,
                                                float* diffStepSize_,
                                                const int rangeMin,
                                                const int rangeMax,
                                                const float* stepSize,
                                                const float gradFactor,
                                                const float beta,
                                                const size_t size)
{
    // For the moment only one step size value per output

    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < size; i += stride) {
        const float fullPrecScale = fullPrecData_[i] / stepSize[0];

        /*****************************Step Size Gradient Computation*************************/
        float qData = fullPrecScale;
        //1st: clip the gradient in interval [rangeMin, rangeMax] and take account of qError
        qData = (qData <= (float) rangeMin ) ? (float) rangeMin :
                (qData >= (float) rangeMax ) ? (float) rangeMax :
                rintf(qData) - qData;

        //2nd: Multiplie backward data with clipped grad
        qData *= diffQuantData_[i];
        //3rd : Multiplie by gradFactor
        qData *= gradFactor;
        if(beta == 0) {
            diffStepSize_[i] = 0.0f;
        }
        diffStepSize_[i] = qData + beta*diffStepSize_[i];
        /**************************************************************************************/

        /*****************************Data/Weights Gradient Computation************************/
        // STE method is simply apply:
        diffFullPrecData_[i] = diffQuantData_[i]*(  (fullPrecScale <= (float) rangeMin ) ? 0.0f :
                                                    (fullPrecScale >= (float) rangeMax ) ? 0.0f :
                                                    1.0f); 
    }
} 


void N2D2::LSQQuantizer_Frame_CUDA_Kernels::cudaF_quantize_back_propagate(float* diffQuantData,
                                                                          const float* fullPrecData, 
                                                                          float* diffFullPrecData,
                                                                          float* diffStepSize,     
                                                                          const int rangeMin,
                                                                          const int rangeMax,
                                                                          const float* stepSize, 
                                                                          const float gradFactor, 
                                                                          const float beta,      
                                                                          const size_t inputSize)
{
    cudaF_LSQ_back_propagate_kernel<<< (inputSize + 255) / 256, 256>>>(diffQuantData,
                                                                       fullPrecData,
                                                                       diffFullPrecData,
                                                                       diffStepSize,
                                                                       rangeMin,
                                                                       rangeMax,
                                                                       stepSize,
                                                                       gradFactor,
                                                                       beta,
                                                                       inputSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
