/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include "Cell/TransposeCell_Frame_CUDA_Kernels.hpp"

template <class T>
__global__
void cudaCopyTranspose_kernel(const T* inputs,
                              T* outputs,
                              unsigned int dimX,
                              unsigned int dimY,
                              unsigned int dimZ,
                              unsigned int dimB,
                              unsigned int permX,
                              unsigned int permY,
                              unsigned int permZ,
                              unsigned int permB)
{
    const unsigned int perms[4] = {permX, permY, permZ, permB};
    const unsigned int dims[4] = {dimX, dimY, dimZ, dimB};
    const unsigned int permDims[4] = {dims[perms[0]], dims[perms[1]],
                                      dims[perms[2]], dims[perms[3]]};

    unsigned int coords[4];
    coords[3] = blockIdx.z;

    for (coords[2] = blockIdx.x; coords[2] < dims[2]; coords[2] += gridDim.x) {
        for (coords[1] = threadIdx.y; coords[1] < dims[1];
            coords[1] += blockDim.y)
        {
            for (coords[0] = threadIdx.x; coords[0] < dims[0];
                coords[0] += blockDim.x)
            {
                const unsigned int iIdx = coords[0]
                    + dims[0] * (coords[1]
                        + dims[1] * (coords[2]
                            + dims[2] * coords[3]));

                const unsigned int oIdx = coords[perms[0]]
                    + permDims[0] * (coords[perms[1]]
                        + permDims[1] * (coords[perms[2]]
                            + permDims[2] * coords[perms[3]]));

                outputs[oIdx] = inputs[iIdx];
            }
        }
    }
}

namespace N2D2 {

template <class T>
void cudaCopyTranspose(const cudaDeviceProp& deviceProp,
                             const T* inputs,
                             T* outputs,
                             unsigned int dimX,
                             unsigned int dimY,
                             unsigned int dimZ,
                             unsigned int dimB,
                             unsigned int permX,
                             unsigned int permY,
                             unsigned int permZ,
                             unsigned int permB)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (dimX * dimY < maxSize)
                                       ? dimX * dimY
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)dimX);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {dimZ, 1, dimB};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaCopyTranspose_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(inputs),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(outputs),
           dimX,
           dimY,
           dimZ,
           dimB,
           permX,
           permY,
           permZ,
           permB);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


template void cudaCopyTranspose(const cudaDeviceProp& deviceProp,
                             const half_float::half* inputs,
                             half_float::half* outputs,
                             unsigned int dimX,
                             unsigned int dimY,
                             unsigned int dimZ,
                             unsigned int dimB,
                             unsigned int permX,
                             unsigned int permY,
                             unsigned int permZ,
                             unsigned int permB);
template void cudaCopyTranspose(const cudaDeviceProp& deviceProp,
                             const float* inputs,
                             float* outputs,
                             unsigned int dimX,
                             unsigned int dimY,
                             unsigned int dimZ,
                             unsigned int dimB,
                             unsigned int permX,
                             unsigned int permY,
                             unsigned int permZ,
                             unsigned int permB);
template void cudaCopyTranspose(const cudaDeviceProp& deviceProp,
                             const double* inputs,
                             double* outputs,
                             unsigned int dimX,
                             unsigned int dimY,
                             unsigned int dimZ,
                             unsigned int dimB,
                             unsigned int permX,
                             unsigned int permY,
                             unsigned int permZ,
                             unsigned int permB);

}
