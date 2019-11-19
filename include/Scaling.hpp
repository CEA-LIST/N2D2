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

#ifndef N2D2_SCALING_H
#define N2D2_SCALING_H

#include <iosfwd>
#include <vector>

#ifdef CUDA
#include "containers/CudaTensor.hpp"
#include "third_party/half.hpp"
#include "Cell/ElemWiseCell_Frame_CUDA_Kernels.hpp" // For cudaSScale
#endif

#include "FloatT.hpp"
#include "ScalingMode.hpp"
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
    FloatingPointScaling(std::vector<Float_T> scalignPerOutput): 
                            mScalingPerOutput(std::move(scalignPerOutput)) {}

    const std::vector<Float_T>& getScalingPerOutput() const {
        return mScalingPerOutput;
    }

    template<typename T>
    void propagate(const Tensor<T>& input, Tensor<T>& output) const {
        std::size_t index = 0;
        for (std::size_t batch = 0; batch < input.dimB(); batch++) {
            for(std::size_t ch = 0; ch < input.dimZ(); ch++) {
                for(std::size_t y = 0; y < input.dimY(); y++) {
                    for(std::size_t x = 0; x < input.dimX(); x++) {
                        output(index) = scale(input(index), ch);
                        index++;
                    }
                }
            }
        }
    }

#ifdef CUDA
    // TODO Optimize
    void propagate(const CudaTensor<double>& /*input*/, CudaTensor<double>& /*output*/) const { 
        throw std::runtime_error("Scaling with double not supported yet.");
    }

    void propagate(const CudaTensor<float>& input, CudaTensor<float>& output) const { 
        for(std::size_t batch = 0; batch < input.dimB(); batch++) {
            for(std::size_t ch = 0; ch < input.dimZ(); ch++) {
                const std::size_t offset = batch*input.dimZ()*input.dimY()*input.dimX() + 
                                           ch*input.dimY()*input.dimX();

                cudaSScale(input.dimY()*input.dimX(),
                           input.getDevicePtr() + offset,
                           mScalingPerOutput[ch], 0.0f, 0.0f,
                           output.getDevicePtr() + offset);
            }
        }
    }

    void propagate(const CudaTensor<half_float::half>& /*input*/, 
                   CudaTensor<half_float::half>& /*output*/) const 
    {
        throw std::runtime_error("Scaling with half floats not supported yet.");
    }
#endif

private:
    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    T scale(T value, std::size_t channel) const {
        // return (T) std::round(value*mScalingPerOutput[channel]);
        return (T) (value*mScalingPerOutput[channel]);
    }

    template<typename T, typename std::enable_if<!std::is_floating_point<T>::value>::type* = nullptr>
    T scale(T value, std::size_t channel) const {
        return (T) std::round(value*mScalingPerOutput[channel]);
    }

private:
    std::vector<Float_T> mScalingPerOutput;
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
 * const std::size_t HALF = 1 << (mScalingPerOutput[o].second - 1);
 * return (data + (data << mScalingPerOutput[o].first) + HALF) >> mScalingPerOutput[o].second;
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

private:
    std::vector<std::pair<unsigned char, unsigned char>> mScalingPerOutput;
};


class Scaling {
public:
    Scaling();

    static Scaling floatingPointScaling(std::vector<Float_T> scalingPerOutput) {
        return Scaling(ScalingMode::FLOAT_MULT, 
                       Utils::make_unique<FloatingPointScaling>(std::move(scalingPerOutput)));
    }

    static Scaling fixedPointScaling(std::size_t nbFractionalBits, 
                                     std::vector<std::int32_t> scalingPerOutput) 
    {
        return Scaling(ScalingMode::FIXED_MULT, 
                       Utils::make_unique<FixedPointScaling>(nbFractionalBits, 
                                                             std::move(scalingPerOutput)));
    }

    static Scaling singleShiftScaling(std::vector<unsigned char> scalingPerOutput) {
        return Scaling(ScalingMode::SINGLE_SHIFT, 
                       Utils::make_unique<SingleShiftScaling>(std::move(scalingPerOutput)));
    }

    static Scaling doubleShiftScaling(std::vector<std::pair<unsigned char, 
                                                            unsigned char>> scalingPerOutput) 
    {
        return Scaling(ScalingMode::DOUBLE_SHIFT, 
                       Utils::make_unique<DoubleShiftScaling>(std::move(scalingPerOutput)));
    }

    ScalingMode getMode() const {
        return mMode;
    }

    const FloatingPointScaling& getFloatingPointScaling() const {
        assert(mMode == ScalingMode::FLOAT_MULT);
        return static_cast<const FloatingPointScaling&>(*mScaling);
    }

    const FixedPointScaling& getFixedPointScaling() const {
        assert(mMode == ScalingMode::FIXED_MULT);
        return static_cast<const FixedPointScaling&>(*mScaling);
    }

    const SingleShiftScaling& getSingleShiftScaling() const {
        assert(mMode == ScalingMode::SINGLE_SHIFT);
        return static_cast<const SingleShiftScaling&>(*mScaling);
    }

    const DoubleShiftScaling& getDoubleShiftScaling() const {
        assert(mMode == ScalingMode::DOUBLE_SHIFT);
        return static_cast<const DoubleShiftScaling&>(*mScaling);
    }

    template<class T>
    void propagate(Tensor<T>& data) const;

    template<class T>
    void propagate(const Tensor<T>& input, Tensor<T>& output) const;

    template<class T>
    void backPropagate(Tensor<T>& data, Tensor<T>& diffData) const;

#ifdef CUDA
    template<class T>
    void propagate(CudaTensor<T>& data) const;

    template<class T>
    void propagate(const CudaTensor<T>& input, CudaTensor<T>& output) const;

    template<class T>
    void backPropagate(CudaTensor<T>& data, CudaTensor<T>& diffData) const;
#endif

private:
    Scaling(ScalingMode mode, std::unique_ptr<AbstractScaling> scaling);

private:
    ScalingMode mMode;
    std::unique_ptr<AbstractScaling> mScaling;
};

template<class T>
inline void Scaling::propagate(Tensor<T>& data) const {
    propagate(data, data);
}

template<class T>
inline void Scaling::propagate(const Tensor<T>& input, Tensor<T>& output) const {
    assert(input.size() == output.size());
    
    switch(mMode) {
        case ScalingMode::NONE:
            break;
        case ScalingMode::FLOAT_MULT:
            static_cast<FloatingPointScaling&>(*mScaling).propagate(input, output);
            break;
        default:
            throw std::runtime_error("Unsupported scaling propagation.");
    }
}

template<class T>
inline void Scaling::backPropagate(Tensor<T>& /*data*/, Tensor<T>& /*diffData*/) const {
    if(mMode == ScalingMode::NONE) {
        return;
    }

    throw std::runtime_error("Unsupported scaling backpropagation.");
}


#ifdef CUDA
template<class T>
inline void Scaling::propagate(CudaTensor<T>& data) const {
    propagate(data, data);
}

template<class T>
inline void Scaling::propagate(const CudaTensor<T>& input, CudaTensor<T>& output) const {
    switch(mMode) {
        case ScalingMode::NONE:
            break;
        case ScalingMode::FLOAT_MULT:
            static_cast<FloatingPointScaling&>(*mScaling).propagate(input, output);
            break;
        default:
            throw std::runtime_error("Unsupported scaling propagation.");
    }
}

template<class T>
inline void Scaling::backPropagate(CudaTensor<T>& /*data*/, CudaTensor<T>& /*diffData*/) const {
    if(mMode == ScalingMode::NONE) {
        return;
    }

    throw std::runtime_error("Unsupported scaling backpropagation.");
}
#endif

}

#endif