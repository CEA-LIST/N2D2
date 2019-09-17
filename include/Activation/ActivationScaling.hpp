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

#ifndef N2D2_ACTIVATION_SCALING_H
#define N2D2_ACTIVATION_SCALING_H

#include <iosfwd>
#include <vector>

#ifdef CUDA
#include "containers/CudaTensor.hpp"
#include "third_party/half.hpp"
#endif

#include "Activation/ActivationScalingMode.hpp"
#include "containers/Tensor.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {

class AbstractScaling {
};

/**
 * Scale value with a floating-point multiplication: 
 * 
 * - return data * mScalingPerOutput[o]; if data is a floating-point
 * - return std::round(data * mScalingPerOutput[o]); if data is an integer
 */
class FloatingPointScaling: public AbstractScaling {
public:
    FloatingPointScaling(std::vector<double> scalignPerOutput): 
                            mScalingPerOutput(std::move(scalignPerOutput)) {}

    const std::vector<double>& getScalingPerOutput() const {
        return mScalingPerOutput;
    }

    template<typename T>
    void propagate(Tensor<T>& data) const {
        std::size_t index = 0;
        for (std::size_t batch = 0; batch < data.dimB(); batch++) {
            for(std::size_t ch = 0; ch < data.dimZ(); ch++) {
                for(std::size_t y = 0; y < data.dimY(); y++) {
                    for(std::size_t x = 0; x < data.dimX(); x++) {
                        data(index) = scale(data(index), ch);
                        index++;
                    }
                }
            }
        }
    }

#ifdef CUDA
    // TODO Optimize
    void propagate(CudaTensor<double>& data) const { 
        for(std::size_t batch = 0; batch < data.dimB(); batch++) {
            for(std::size_t output = 0; output < data.dimZ(); output++) {
                const double alpha = mScalingPerOutput[output];
                CHECK_CUBLAS_STATUS(
                    cublasDscal(CudaContext::cublasHandle(), data.dimY()*data.dimX(),
                                &alpha, data.getDevicePtr() + 
                                        (output*data.dimY()*data.dimX() + 
                                        batch*data.dimZ()*data.dimY()*data.dimX()), 1)
                );
            }
        }
    }

    void propagate(CudaTensor<float>& data) const { 
        for(std::size_t batch = 0; batch < data.dimB(); batch++) {
            for(std::size_t output = 0; output < data.dimZ(); output++) {
                const float alpha = static_cast<float>(mScalingPerOutput[output]);
                CHECK_CUBLAS_STATUS(
                    cublasSscal(CudaContext::cublasHandle(), data.dimY()*data.dimX(),
                                &alpha, data.getDevicePtr() + 
                                        (output*data.dimY()*data.dimX() + 
                                        batch*data.dimZ()*data.dimY()*data.dimX()), 1)
                );
            }
        }
    }

    void propagate(CudaTensor<half_float::half>& /*data*/) const {
        throw std::runtime_error("Rescaling not supported with half floats.");
    }
#endif

private:
    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    T scale(T value, std::size_t channel) const {
        return (T) (value*mScalingPerOutput[channel]);
    }

    template<typename T, typename std::enable_if<!std::is_floating_point<T>::value>::type* = nullptr>
    T scale(T value, std::size_t channel) const {
        return (T) std::round(value*mScalingPerOutput[channel]);
    }

private:
    std::vector<double> mScalingPerOutput;
};

/**
 * Scale value with a fixed-point multiplication: 
 * 
 * const std::size_t HALF = 1 << (mNbFractionalBits - 1);
 * return (data * mScalingPerOutput[o] + HALF) >> mNbFractionalBits;
 * 
 * TODO Implement scaling and improve documentation.
 */
class FixedPointScaling: public AbstractScaling {
public:
    static const std::size_t DEFAULT_NB_FRACTIONAL_BITS = 30;

    FixedPointScaling(std::size_t nbFractionalBits, std::vector<std::int32_t> scaling)
           : mNbFractionalBits(nbFractionalBits), mScalingPerOutput(std::move(scaling))
    {}

    const std::vector<std::int32_t>& getScalingPerOutput() const {
        return mScalingPerOutput;
    }

    std::size_t getFractionalBits() const {
        return mNbFractionalBits;
    }

private:
    std::size_t mNbFractionalBits;
    std::vector<std::int32_t> mScalingPerOutput;
};


/**
 * Scale value with a single shift: 
 * 
 * const std::size_t HALF = 1 << (mScalingPerOutput[o] - 1);
 * return (data + HALF) >> mScalingPerOutput[o];
 * 
 * TODO Implement scaling and improve documentation.
 */
class SingleShiftScaling: public AbstractScaling {
public:
    SingleShiftScaling(std::vector<unsigned char> scaling): mScalingPerOutput(std::move(scaling)) {}

    const std::vector<unsigned char>& getScalingPerOutput() const {
        return mScalingPerOutput;
    }

private:
    std::vector<unsigned char> mScalingPerOutput;
};

/**
 * Scale values with a double shift: 
 * 
 * const std::size_t HALF = 1 << (mScalingPerOutput[o].first - 1);
 * return (data + (data >> mScalingPerOutput[o].second) + HALF) >> mScalingPerOutput[o].first;
 * 
 * TODO Implement scaling and improve documentation.
 */
class DoubleShiftScaling: public AbstractScaling {
public:
    DoubleShiftScaling(std::vector<std::pair<unsigned char, unsigned char>> scaling)
                                : mScalingPerOutput(std::move(scaling)) {}

    const std::vector<std::pair<unsigned char, unsigned char>>& getScalingPerOutput() const {
        return mScalingPerOutput;
    }

public:
    static const unsigned char NO_SHIFT = std::numeric_limits<unsigned char>::max();

private:
    std::vector<std::pair<unsigned char, unsigned char>> mScalingPerOutput;
};


class ActivationScaling {
public:
    ActivationScaling();

    static ActivationScaling floatingPointScaling(std::vector<double> scalingPerOutput) {
        return ActivationScaling(ActivationScalingMode::FLOAT_MULT, 
                                 Utils::make_unique<FloatingPointScaling>(std::move(scalingPerOutput)));
    }

    static ActivationScaling fixedPointScaling(std::size_t nbFractionalBits, 
                                               std::vector<std::int32_t> scalingPerOutput) 
    {
        return ActivationScaling(ActivationScalingMode::FIXED_MULT, 
                                 Utils::make_unique<FixedPointScaling>(nbFractionalBits, 
                                                                       std::move(scalingPerOutput)));
    }

    static ActivationScaling singleShiftScaling(std::vector<unsigned char> scalingPerOutput) {
        return ActivationScaling(ActivationScalingMode::SINGLE_SHIFT, 
                                 Utils::make_unique<SingleShiftScaling>(std::move(scalingPerOutput)));
    }

    static ActivationScaling doubleShiftScaling(std::vector<std::pair<unsigned char, 
                                                                      unsigned char>> scalingPerOutput) 
    {
        return ActivationScaling(ActivationScalingMode::DOUBLE_SHIFT, 
                                 Utils::make_unique<DoubleShiftScaling>(std::move(scalingPerOutput)));
    }

    ActivationScalingMode getMode() const {
        return mMode;
    }

    const FloatingPointScaling& getFloatingPointScaling() const {
        assert(mMode == FLOAT_MULT);
        return static_cast<const FloatingPointScaling&>(*mScaling);
    }

    const FixedPointScaling& getFixedPointScaling() const {
        assert(mMode == FIXED_MULT);
        return static_cast<const FixedPointScaling&>(*mScaling);
    }

    const SingleShiftScaling& getSingleShiftScaling() const {
        assert(mMode == SINGLE_SHIFT);
        return static_cast<const SingleShiftScaling&>(*mScaling);
    }

    const DoubleShiftScaling& getDoubleShiftScaling() const {
        assert(mMode == DOUBLE_SHIFT);
        return static_cast<const DoubleShiftScaling&>(*mScaling);
    }

    template<class T>
    void propagate(Tensor<T>& data) const;

    template<class T>
    void backPropagate(Tensor<T>& data, Tensor<T>& diffData) const;

#ifdef CUDA
    template<class T>
    void propagate(CudaTensor<T>& data) const;

    template<class T>
    void backPropagate(CudaTensor<T>& data, CudaTensor<T>& diffData) const;
#endif

private:
    ActivationScaling(ActivationScalingMode mode, std::unique_ptr<AbstractScaling> scaling);

private:
    ActivationScalingMode mMode;
    std::unique_ptr<AbstractScaling> mScaling;
};

template<class T>
inline void ActivationScaling::propagate(Tensor<T>& data) const {
    switch(mMode) {
        case ActivationScalingMode::NONE:
            break;
        case ActivationScalingMode::FLOAT_MULT:
            static_cast<FloatingPointScaling&>(*mScaling).propagate(data);
            break;
        default:
            throw std::runtime_error("Unsupported scaling propagation.");
    }
}

template<class T>
inline void ActivationScaling::backPropagate(Tensor<T>& /*data*/, Tensor<T>& /*diffData*/) const {
    if(mMode == ActivationScalingMode::NONE) {
        return;
    }

    throw std::runtime_error("Backpropagation of activation scaling not supported yet.");
}


#ifdef CUDA
template<class T>
inline void ActivationScaling::propagate(CudaTensor<T>& data) const {
    switch(mMode) {
        case ActivationScalingMode::NONE:
            break;
        case ActivationScalingMode::FLOAT_MULT:
            static_cast<FloatingPointScaling&>(*mScaling).propagate(data);
            break;
        default:
            throw std::runtime_error("Unsupported scaling.");
    }
}

template<class T>
inline void ActivationScaling::backPropagate(CudaTensor<T>& /*data*/, CudaTensor<T>& /*diffData*/) const {
    if(mMode == ActivationScalingMode::NONE) {
        return;
    }

    throw std::runtime_error("Backpropagation of activation scaling not supported yet.");
}
#endif

}

#endif