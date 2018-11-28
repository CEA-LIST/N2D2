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

#ifndef N2D2_SGDSOLVER_H
#define N2D2_SGDSOLVER_H

#include "Solver/Solver.hpp"

namespace N2D2 {
class SGDSolver : public Solver {
public:
    typedef std::function<std::shared_ptr<SGDSolver>()> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static const char* Type;

    enum LearningRatePolicy {
        None,
        StepDecay,
        ExponentialDecay,
        InvTDecay,
        PolyDecay,
        InvDecay
    };

    SGDSolver();
    const char* getType() const
    {
        return Type;
    };
    SGDSolver(const SGDSolver& solver);
    std::shared_ptr<SGDSolver> clone() const
    {
        return std::shared_ptr<SGDSolver>(doClone());
    }
    bool isNewIteration() const
    {
        return (mIterationPass == 0);
    }
    std::pair<double, double> getRange() const
    {
        return std::make_pair(mMinVal, mMaxVal);
    }
    std::pair<double, double> getQuantizedRange() const
    {
        return std::make_pair(mMinValQuant, mMaxValQuant);
    }
    virtual ~SGDSolver() {};

protected:
    double getLearningRate(unsigned int batchSize);
    virtual void saveInternal(std::ostream& state, std::ostream& log) const;
    virtual void loadInternal(std::istream& state);

    /// Initial learning rate
    Parameter<double> mLearningRate;
    /// Momentum
    Parameter<double> mMomentum;
    /// Decay
    Parameter<double> mDecay;
    /// Power
    Parameter<double> mPower;
    /// Global batch size = batch size * mIterationSize
    Parameter<unsigned int> mIterationSize;
    /// MaxIterations
    Parameter<unsigned long long int> mMaxIterations;
    /// Learning rate decay policy
    Parameter<LearningRatePolicy> mLearningRatePolicy;
    /// Learning rate step size
    Parameter<unsigned int> mLearningRateStepSize;
    /// Learning rate decay
    Parameter<double> mLearningRateDecay;
    /// Quantization levels (0 = no quantization)
    Parameter<unsigned int> mQuantizationLevels;
    /// If true, don't clamp the weights between -1 and 1 during learning
    Parameter<bool> mClamping;

    unsigned int mIterationPass;
    unsigned int mNbIterations;
    double mMinVal;
    double mMaxVal;
    double mMinValQuant;
    double mMaxValQuant;

private:
    virtual SGDSolver* doClone() const = 0;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::SGDSolver::LearningRatePolicy>::data[]
    = {"None",
       "StepDecay",
       "ExponentialDecay",
       "InvTDecay",
       "PolyDecay",
       "InvDecay"};
}

#endif // N2D2_SGDSOLVER_H
