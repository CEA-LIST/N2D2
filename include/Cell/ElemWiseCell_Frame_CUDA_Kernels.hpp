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
void cudaUZeroInit(unsigned int size,
                   unsigned int* data);
void cudaSZeroInit(unsigned int size,
                   float* data);
void cudaSSqrt(unsigned int size,
               float* data);
void cudaSMult(unsigned int size,
               float* a,
               float* b,
               const float beta,
               float* result);
void cudaSScale(unsigned int size,
                float* input,
                const float scale,
                const float beta,
                float* result);
void cudaSScaleAbs(unsigned int size,
                   float* input,
                   const float scale,
                   const float beta,
                   float* result);
void cudaSScaleSign(unsigned int size,
                    float* input,
                    float* sign,
                    const float scale,
                    const float beta,
                    float* result);
void cudaSScaleSquare(unsigned int size,
                      float* input,
                      const float scale,
                      const float beta,
                      float* result);
void cudaSMaxForward(unsigned int size,
               float* input,
               float* maxVal,
               const unsigned int idx,
               unsigned int* argMax);
void cudaSMaxBackward(unsigned int size,
                      float* diffInput,
                      const unsigned int idx,
                      unsigned int* argMax,
                      const float beta,
                      float* result);
void cudaSEuclideanSumBackward(unsigned int size,
                               float* diffInput,
                               float* input,
                               float* output,
                               const float scale,
                               const float beta,
                               float* result);
}

#endif // N2D2_ELEMWISECELL_FRAME_CUDA_KERNELS_H
