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

#ifndef N2D2_SCALING_CUDA_KERNELS_H
#define N2D2_SCALING_CUDA_KERNELS_H

#include "FloatT.hpp"
#include <utility>


namespace N2D2 {

template<typename T>
void cudaFloatingPointScaling_propagate(const T* input, T* output,
                                        std::size_t batchSize, std::size_t nbChannels,
                                        std::size_t heigth, std::size_t width,
                                        bool isClipped,
                                        Float_T* clippingFactorPerChannel,
                                        Float_T* scalingFactorPerChannel,
                                        std::size_t quantizedNbBits, bool isOutputUnsigned,
                                        dim3 blocksPerGrid, dim3 threadsPerBlock);

template<typename T>
void cudaFixedPointScaling_propagate(const T* input, T* output,
                                     std::size_t batchSize, std::size_t nbChannels,
                                     std::size_t heigth, std::size_t width,
                                     bool isClipped,
                                     Float_T* clippingFactorPerChannel,
                                     std::int32_t* scalingPerOutput, std::size_t nbFractionalBits,
                                     std::size_t quantizedNbBits, bool isOutputUnsigned,
                                     dim3 blocksPerGrid, dim3 threadsPerBlock);

template<typename T>
void cudaSingleShiftScaling_propagate(const T* input, T* output,
                                      std::size_t batchSize, std::size_t nbChannels,
                                      std::size_t heigth, std::size_t width,
                                      bool isClipped,
                                      Float_T* clippingFactorPerChannel,
                                      unsigned char* scalingPerOutput,
                                      std::size_t quantizedNbBits, bool isOutputUnsigned,
                                      dim3 blocksPerGrid, dim3 threadsPerBlock);

template<typename T>
void cudaDoubleShiftScaling_propagate(const T* input, T* output,
                                      std::size_t batchSize, std::size_t nbChannels,
                                      std::size_t heigth, std::size_t width,
                                      bool isClipped,
                                      std::pair<unsigned char, unsigned char>* clippingFactorPerChannel,
                                      std::pair<unsigned char, unsigned char>* scalingPerOutput,
                                      std::size_t quantizedNbBits, bool isOutputUnsigned,
                                      dim3 blocksPerGrid, dim3 threadsPerBlock);

}

#endif
