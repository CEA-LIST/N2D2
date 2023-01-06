/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#include "Quantizer/QAT/Kernel/PruneQuantizer_Frame_CUDA_Kernels.hpp"
#include "CudaUtils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void
cudaPruneMasks_kernel(__half* srcData,
                      __half* dstData,
                      unsigned int* masks,
                      unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dstData[i] = srcData[i] * __uint2half_ru(masks[i]);
    }
}

void N2D2::PruneQuantizer_Frame_CUDA_Kernels::apply_pruning_with_masks_H(half_float::half* data,
                                                                         half_float::half* dataPruned,
                                                                         unsigned int* masks,
                                                                         unsigned int size)
{
    cudaPruneMasks_kernel<<<(size + 255) / 256, 256>>>(reinterpret_cast<__half*>(data), 
                                                       reinterpret_cast<__half*>(dataPruned),
                                                       masks, 
                                                       size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
} 


void N2D2::PruneQuantizer_Frame_CUDA_Kernels::apply_pruning_with_masks_F(float* data,
                                                                         float* dataPruned,
                                                                         unsigned int* masks,
                                                                         unsigned int size)
{
    thrust::device_ptr<float> dataPtr(data);
    thrust::device_ptr<float> dataPrunedPtr(dataPruned);
    thrust::device_ptr<unsigned int> masksPtr(masks);
    thrust::transform(dataPtr, dataPtr + size, masksPtr, dataPrunedPtr, thrust::multiplies<float>());
}    


void N2D2::PruneQuantizer_Frame_CUDA_Kernels::apply_pruning_with_masks_D(double* data,
                                                                         double* dataPruned,
                                                                         unsigned int* masks,
                                                                         unsigned int size)
{
    thrust::device_ptr<double> dataPtr(data);
    thrust::device_ptr<double> dataPrunedPtr(dataPruned);
    thrust::device_ptr<unsigned int> masksPtr(masks);
    thrust::transform(dataPtr, dataPtr + size, masksPtr, dataPrunedPtr, thrust::multiplies<double>());
} 
