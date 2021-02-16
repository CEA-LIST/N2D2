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

#include <cassert>
#include "Scaling_Kernels.hpp"
#include "containers/Tensor.hpp"
#include "third_party/half.hpp"
#include "utils/Utils.hpp"


template<typename T>
T saturate(T value, std::size_t quantizedNbBits, bool isOutputUnsigned) {
    assert(quantizedNbBits > 0);
    
    const T min = isOutputUnsigned?0:
                                  -(1ll << (quantizedNbBits - 1ll));
    const T max = isOutputUnsigned?(1ll << quantizedNbBits) - 1ll:
                                   (1ll << (quantizedNbBits - 1ll)) - 1ll;

    return N2D2::Utils::clamp(value, min, max);
}

template<typename T>
void N2D2::floatingPointScaling_propagate(const Tensor<T>& input, Tensor<T>& output,
                                          std::size_t batchSize, std::size_t nbChannels,
                                          std::size_t heigth, std::size_t width,
                                          bool /*isClipped*/,
                                          const std::vector<Float_T>& /*clippingFactorPerChannel*/,
                                          const std::vector<Float_T>& scalingFactorPerChannel,
                                          std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    std::size_t index = 0;
    for (std::size_t batch = 0; batch < batchSize; batch++) {
        for(std::size_t ch = 0; ch < nbChannels; ch++) {
            for(std::size_t y = 0; y < heigth; y++) {
                for(std::size_t x = 0; x < width; x++) {

                    //TODO::add clipping here properly, nothing for now
                    auto res = input(index)*scalingFactorPerChannel[ch];
                    if(quantizedNbBits > 0) {
                        res = saturate(std::round(res), quantizedNbBits, isOutputUnsigned);
                    }


                    output(index) = (T) res;
                    index++;
                }
            }
        }
    }
}

template<typename T>
void N2D2::fixedPointScaling_propagate(const Tensor<T>& input, Tensor<T>& output,
                                       std::size_t batchSize, std::size_t nbChannels,
                                       std::size_t heigth, std::size_t width,
                                       bool /*isClipped*/,
                                       const std::vector<std::int32_t>& /*clippingFactorPerChannel*/,
                                       const std::vector<std::int32_t>& scalingFactorPerChannel, 
                                       std::size_t nbFractionalBits,
                                       std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    if(quantizedNbBits == 0) {
        throw std::runtime_error("FixedPointScaling::propagate can only be used on quantized network.");
    }
    
    std::size_t index = 0;
    for (std::size_t batch = 0; batch < batchSize; batch++) {
        for(std::size_t ch = 0; ch < nbChannels; ch++) {
            for(std::size_t y = 0; y < heigth; y++) {
                for(std::size_t x = 0; x < width; x++) {
                     //TODO::add clipping here properly, nothing for now
                    const long long half = (nbFractionalBits > 0)
                        ? (1ll << (nbFractionalBits - 1))
                        : 0ll;
                    const long long val = static_cast<long long>(std::round(input(index)));
                    const long long res = (val*scalingFactorPerChannel[ch] + half) >> nbFractionalBits;


                    output(index) = saturate(res, quantizedNbBits, isOutputUnsigned);
                    index++;
                }
            }
        }
    }
}

template<typename T>
void N2D2::singleShiftScaling_propagate(const Tensor<T>& input, Tensor<T>& output,
                                        std::size_t batchSize, std::size_t nbChannels,
                                        std::size_t heigth, std::size_t width,
                                        bool /*isClipped*/,
                                        const std::vector<unsigned char>& /*clippingFactorPerChannel*/,
                                        const std::vector<unsigned char>& scalingFactorPerChannel,
                                        std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    if(quantizedNbBits == 0) {
        throw std::runtime_error("SingleShiftScaling::propagate can only be used on quantized network.");
    }
    
    std::size_t index = 0;
    for (std::size_t batch = 0; batch < batchSize; batch++) {
        for(std::size_t ch = 0; ch < nbChannels; ch++) {
            for(std::size_t y = 0; y < heigth; y++) {
                for(std::size_t x = 0; x < width; x++) {
                    //TODO::add clipping here properly, nothing for now
                    const long long half = (scalingFactorPerChannel[ch] > 0)
                        ? (1ll << (scalingFactorPerChannel[ch] - 1ll))
                        : 0ll;
                    const long long val = static_cast<long long>(std::round(input(index)));
                    const long long res = (val + half) >> scalingFactorPerChannel[ch];


                    output(index) = saturate(res, quantizedNbBits, isOutputUnsigned);
                    index++;
                }
            }
        }
    }
}

template<typename T>
void N2D2::doubleShiftScaling_propagate(const Tensor<T>& input, Tensor<T>& output,
                std::size_t batchSize, std::size_t nbChannels,
                std::size_t heigth, std::size_t width,
                bool /*isClipped*/,
                const std::vector<std::pair<unsigned char, unsigned char>>& /*clippingFactorPerChannel*/,
                const std::vector<std::pair<unsigned char, unsigned char>>& scalingFactorPerChannel,
                std::size_t quantizedNbBits, bool isOutputUnsigned)
{
    if(quantizedNbBits == 0) {
        throw std::runtime_error("DoubleShiftScaling::propagate can only be used on quantized network.");
    }
    
    std::size_t index = 0;
    for (std::size_t batch = 0; batch < batchSize; batch++) {
        for(std::size_t ch = 0; ch < nbChannels; ch++) {
            for(std::size_t y = 0; y < heigth; y++) {
                for(std::size_t x = 0; x < width; x++) {
                    //TODO::add clipping here properly, nothing for now
                    const long long half = (scalingFactorPerChannel[ch].second > 0)
                        ? (1ll << (scalingFactorPerChannel[ch].second - 1ll))
                        : 0ll;
                    const long long val = static_cast<long long>(std::round(input(index)));
                    const long long res = (
                        val + (val << scalingFactorPerChannel[ch].first) +  half
                    ) >> scalingFactorPerChannel[ch].second;


                    output(index) = saturate(res, quantizedNbBits, isOutputUnsigned);
                    index++;
                }
            }
        }
    }
}






template void N2D2::floatingPointScaling_propagate<float>(const N2D2::Tensor<float>& input, N2D2::Tensor<float>& output,
                                                          std::size_t batchSize, std::size_t nbChannels,
                                                          std::size_t heigth, std::size_t width,
                                                          bool /*isClipped*/,
                                                          const std::vector<Float_T>& /*clippingFactorPerChannel*/,
                                                          const std::vector<Float_T>& scalingFactorPerChannel,
                                                          std::size_t quantizedNbBits, bool isOutputUnsigned);

template void N2D2::floatingPointScaling_propagate<double>(const N2D2::Tensor<double>& input, N2D2::Tensor<double>& output,
                                                           std::size_t batchSize, std::size_t nbChannels,
                                                           std::size_t heigth, std::size_t width,
                                                           bool /*isClipped*/,
                                                           const std::vector<Float_T>& /*clippingFactorPerChannel*/,
                                                           const std::vector<Float_T>& scalingFactorPerChannel,
                                                           std::size_t quantizedNbBits, bool isOutputUnsigned);

template void N2D2::floatingPointScaling_propagate<half_float::half>(const N2D2::Tensor<half_float::half>& input, N2D2::Tensor<half_float::half>& output,
                                                                     std::size_t batchSize, std::size_t nbChannels,
                                                                     std::size_t heigth, std::size_t width,
                                                                     bool /*isClipped*/,
                                                                     const std::vector<Float_T>& /*clippingFactorPerChannel*/,
                                                                     const std::vector<Float_T>& scalingFactorPerChannel,
                                                                     std::size_t quantizedNbBits, bool isOutputUnsigned);


template void N2D2::fixedPointScaling_propagate<float>(const N2D2::Tensor<float>& input, N2D2::Tensor<float>& output,
                                                       std::size_t batchSize, std::size_t nbChannels,
                                                       std::size_t heigth, std::size_t width,
                                                       bool /*isClipped*/,
                                                       const std::vector<std::int32_t>& /*clippingFactorPerChannel*/,
                                                       const std::vector<std::int32_t>& scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                       std::size_t quantizedNbBits, bool isOutputUnsigned);
template void N2D2::fixedPointScaling_propagate<double>(const N2D2::Tensor<double>& input, N2D2::Tensor<double>& output,
                                                        std::size_t batchSize, std::size_t nbChannels,
                                                        std::size_t heigth, std::size_t width,
                                                        bool /*isClipped*/,
                                                        const std::vector<std::int32_t>& /*clippingFactorPerChannel*/,
                                                        const std::vector<std::int32_t>& scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                        std::size_t quantizedNbBits, bool isOutputUnsigned);
template void N2D2::fixedPointScaling_propagate<half_float::half>(const N2D2::Tensor<half_float::half>& input, N2D2::Tensor<half_float::half>& output,
                                                                  std::size_t batchSize, std::size_t nbChannels,
                                                                  std::size_t heigth, std::size_t width,
                                                                  bool /*isClipped*/,
                                                                  const std::vector<std::int32_t>& /*clippingFactorPerChannel*/,
                                                                  const std::vector<std::int32_t>& scalingFactorPerChannel, std::size_t nbFractionalBits,
                                                                  std::size_t quantizedNbBits, bool isOutputUnsigned);


template void N2D2::singleShiftScaling_propagate<float>(const N2D2::Tensor<float>& input, N2D2::Tensor<float>& output,
                                                        std::size_t batchSize, std::size_t nbChannels,
                                                        std::size_t heigth, std::size_t width,
                                                        bool /*isClipped*/,
                                                        const std::vector<unsigned char>& /*clippingFactorPerChannel*/,
                                                        const std::vector<unsigned char>& scalingFactorPerChannel,
                                                        std::size_t quantizedNbBits, bool isOutputUnsigned);
template void N2D2::singleShiftScaling_propagate<double>(const N2D2::Tensor<double>& input, N2D2::Tensor<double>& output,
                                                         std::size_t batchSize, std::size_t nbChannels,
                                                         std::size_t heigth, std::size_t width,
                                                         bool /*isClipped*/,
                                                         const std::vector<unsigned char>& /*clippingFactorPerChannel*/,
                                                         const std::vector<unsigned char>& scalingFactorPerChannel,
                                                         std::size_t quantizedNbBits, bool isOutputUnsigned);
template void N2D2::singleShiftScaling_propagate<half_float::half>(const N2D2::Tensor<half_float::half>& input, N2D2::Tensor<half_float::half>& output,
                                                                   std::size_t batchSize, std::size_t nbChannels,
                                                                   std::size_t heigth, std::size_t width,
                                                                   bool /*isClipped*/,
                                                                   const std::vector<unsigned char>& /*clippingFactorPerChannel*/,
                                                                   const std::vector<unsigned char>& scalingFactorPerChannel,
                                                                   std::size_t quantizedNbBits, bool isOutputUnsigned);


template void N2D2::doubleShiftScaling_propagate<float>(const N2D2::Tensor<float>& input, N2D2::Tensor<float>& output,
                                                            std::size_t batchSize, std::size_t nbChannels,
                                                            std::size_t heigth, std::size_t width,
                                                            bool /*isClipped*/,
                                                            const std::vector<std::pair<unsigned char, unsigned char>>& /*clippingFactorPerChannel*/,
                                                            const std::vector<std::pair<unsigned char, unsigned char>>& scalingFactorPerChannel,
                                                            std::size_t quantizedNbBits, bool isOutputUnsigned);
template void N2D2::doubleShiftScaling_propagate<double>(const N2D2::Tensor<double>& input, N2D2::Tensor<double>& output,
                                                             std::size_t batchSize, std::size_t nbChannels,
                                                             std::size_t heigth, std::size_t width,
                                                             bool /*isClipped*/,
                                                             const std::vector<std::pair<unsigned char, unsigned char>>& /*clippingFactorPerChannel*/,
                                                             const std::vector<std::pair<unsigned char, unsigned char>>& scalingFactorPerChannel,
                                                             std::size_t quantizedNbBits, bool isOutputUnsigned);
template void N2D2::doubleShiftScaling_propagate<half_float::half>(const N2D2::Tensor<half_float::half>& input, N2D2::Tensor<half_float::half>& output,
                                                                       std::size_t batchSize, std::size_t nbChannels,
                                                                       std::size_t heigth, std::size_t width,
                                                                       bool /*isClipped*/,
                                                                       const std::vector<std::pair<unsigned char, unsigned char>>& /*clippingFactorPerChannel*/,
                                                                       const std::vector<std::pair<unsigned char, unsigned char>>& scalingFactorPerChannel,
                                                                       std::size_t quantizedNbBits, bool isOutputUnsigned);
