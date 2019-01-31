/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_ACTIVATION_H
#define N2D2_ACTIVATION_H

#include "utils/Parameterizable.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {

class BaseTensor;

class Activation : public Parameterizable {
public:
    enum MovingAverageType {
        WMA,
        EMA
    };

    Activation();
    virtual const char* getType() const = 0;
    virtual void propagate(BaseTensor& data, bool inference = false) = 0;
    virtual void backPropagate(BaseTensor& data, BaseTensor& diffData) = 0;
    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);
    void setPreQuantizeScaling(double scaling);
    virtual ~Activation() {};

protected:
    virtual void saveInternal(std::ostream& /*state*/,
                              std::ostream& /*log*/) const {};
    virtual void loadInternal(std::istream& /*state*/) {};

    /// Shifting
    Parameter<int> mShifting;
    /// Quantization levels (0 = no quantization)
    Parameter<unsigned int> mQuantizationLevels;
    /// Number of steps before quantization starts
    Parameter<unsigned int> mQuantizationDelay;
    /// Moving average type, used for quantization
    Parameter<MovingAverageType> mMovingAverage;
    /// Moving average window for WMA
    /// and EMA with alpha = 2/(N + 1) if mEMA_Alpha = 0.0
    Parameter<unsigned int> mMA_Window;
    /// EMA coefficient: should be close to 0 to smooth across many samples
    /// If mEMA_Alpha = 0.0, the value 2/(mMA_Window + 1) is used
    Parameter<double> mEMA_Alpha;
    /// Rounding to the nearest INT in log2:
    /// Rounding rate, between 0.0 (no rounding) to 1.0
    Parameter<double> mLog2RoundingRate;
    /// Rounding power, or progressivity,
    /// from 0.0 (no progressivity) to 1.0 or more (progressive)
    Parameter<double> mLog2RoundingPower;

    unsigned long long int mNbSteps;
    double mPreQuantizeScaling;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::Activation::MovingAverageType>::data[]
    = {"WMA", "EMA"};
}

#endif // N2D2_ACTIVATION_H
