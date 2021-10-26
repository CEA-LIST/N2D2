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

#ifndef N2D2_DISTANCECELL_FRAME_CUDA_KERNELS_H
#define N2D2_DISTANCECELL_FRAME_CUDA_KERNELS_H

#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "CudaUtils.hpp"
#include "third_party/half.hpp"

namespace N2D2 {

template <class T>
void cudaDistanceL2Forward(unsigned int size,
                            unsigned int nb_class, 
                            unsigned int feat_dim,
                            const T *inputs,
                            const T *means,
                            //const T *sigma,
                            T *dist,
                            T *outputs);

template <class T>
void cudaMargin(unsigned int size, 
                const T *label, 
                const T margin,
                T* outputs);

template <class T>
void cudaDistanceL2Backward_input(unsigned int size,
                                    unsigned int nb_class, 
                                    unsigned int feat_dim,
                                    const T margin,
                                    const T centercoef,
                                    const T *label,
                                    const T *inputs,
                                    const T *means, 
                                    const T *, 
                                    T *diffOutputs);

template <class T>           
void cudaDistanceL2Backward_mean(unsigned int size,
                                    unsigned int nb_class, 
                                    unsigned int feat_dim,
                                    unsigned int batchsize,
                                    const T margin,
                                    const T centercoef,
                                    const T *label,
                                    const T *inputs,
                                    const T *means, 
                                    const T *diffInputs, 
                                    T *diffMeans);

}

#endif // N2D2_DISTANCECELL_FRAME_CUDA_KERNELS_H
