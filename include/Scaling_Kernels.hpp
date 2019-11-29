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

#ifndef N2D2_SCALING_KERNELS_H
#define N2D2_SCALING_KERNELS_H

#include "FloatT.hpp"
#include <utility>


namespace N2D2 {

template<class T>
class Tensor;

template<typename T>
void floatingPointScaling_propagate(const Tensor<T>& input, Tensor<T>& output,
                                    std::size_t batchSize, std::size_t nbChannels,
                                    std::size_t heigth, std::size_t width,
                                    const std::vector<Float_T>& scalingFactorPerChannel,
                                    std::size_t quantizedNbBits, bool isOutputUnsigned);

template<typename T>
void fixedPointScaling_propagate(const Tensor<T>& input, Tensor<T>& output,
                                 std::size_t batchSize, std::size_t nbChannels,
                                 std::size_t heigth, std::size_t width,
                                 const std::vector<std::int32_t>& scalingPerOutput, 
                                 std::size_t nbFractionalBits,
                                 std::size_t quantizedNbBits, bool isOutputUnsigned);

template<typename T>
void singleShiftScaling_propagate(const Tensor<T>& input, Tensor<T>& output,
                                  std::size_t batchSize, std::size_t nbChannels,
                                  std::size_t heigth, std::size_t width,
                                  const std::vector<unsigned char>& scalingPerOutput,
                                  std::size_t quantizedNbBits, bool isOutputUnsigned);

template<typename T>
void doubleShiftScaling_propagate(const Tensor<T>& input, Tensor<T>& output,
                        std::size_t batchSize, std::size_t nbChannels,
                        std::size_t heigth, std::size_t width,
                        const std::vector<std::pair<unsigned char, unsigned char>>& scalingPerOutput,
                        std::size_t quantizedNbBits, bool isOutputUnsigned);

}

#endif
