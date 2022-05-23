/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_ELEMWISECELL_FRAME_CUDA_KERNELS_H
#define N2D2_ELEMWISECELL_FRAME_CUDA_KERNELS_H

#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaUtils.hpp"

namespace N2D2 {
template <class T>
void cudaZeroInit(unsigned int size,
                   T* data);
template <class T>
void cudaSqrt(unsigned int size,
               T* data);
template <class T>
void cudaMult(unsigned int size,
               T* a,
               T* b,
               const T beta,
               T* result);
template <class T>
void cudaMultBroadcast(unsigned int sizeMax,
              unsigned int mapSize,
              unsigned int nbOutputs,
              unsigned int batchSize,
              unsigned int sizeA,
              unsigned int sizeB,
                     T* a,
                     T* b,
                     const T beta,
                     T* result);
template <class T>
void cudaReducePerKernel(T* inputToReduce,
        T* result,
        unsigned int nbIter,
        unsigned int kernelSize);
template <class T>
void cudaScale(unsigned int size,
                T* input,
                const T scale,
                const T shift,
                const T beta,
                T* result);
template <class T>
void cudaScaleAbs(unsigned int size,
                   T* input,
                   const T scale,
                   const T beta,
                   T* result);
template <class T>
void cudaScaleSign(unsigned int size,
                    T* input,
                    T* sign,
                    const T scale,
                    const T beta,
                    T* result);
template <class T>
void cudaScaleSquare(unsigned int size,
                      T* input,
                      const T scale,
                      const T shift,
                      const T beta,
                      T* result);
template <class T>
void cudaMaxForward(unsigned int size,
               T* input,
               T* maxVal,
               const unsigned int idx,
               unsigned int* argMax);
template <class T>
void cudaMaxBackward(unsigned int size,
                      T* diffInput,
                      const unsigned int idx,
                      unsigned int* argMax,
                      const T beta,
                      T* result);
template <class T>
void cudaEuclideanSumBackward(unsigned int size,
                               T* diffInput,
                               T* input,
                               T* output,
                               const T scale,
                               const T beta,
                               T* result);
}

#endif // N2D2_ELEMWISECELL_FRAME_CUDA_KERNELS_H
