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

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@?$SGDSolver_Frame@M@N2D2@@0U?$Registrar@V?$SGDSolver@M@N2D2@@@2@A")
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@?$SGDSolver_Frame_CUDA@M@N2D2@@0U?$Registrar@V?$SGDSolver@M@N2D2@@@2@A")
#endif
#endif

namespace N2D2 {
template <class T> class SGDSolver : public Solver<T> {
public:
    typedef std::function<std::shared_ptr<SGDSolver<T> >()> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    enum LearningRatePolicy {
        None,
        StepDecay,
        ExponentialDecay,
        InvTDecay,
        PolyDecay,
        InvDecay
    };

    SGDSolver();
    std::shared_ptr<SGDSolver<T> > clone() const
    {
        return std::shared_ptr<SGDSolver<T> >(doClone());
    }
    bool isNewIteration() const
    {
        return (mIterationPass == 0);
    }
    virtual ~SGDSolver() {};

protected:
    T getLearningRate(unsigned int batchSize);

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
    /// If true, don't clamp the weights between -1 and 1 during learning
    Parameter<bool> mClamping;

    unsigned int mIterationPass;
    unsigned int mNbIterations;

private:
    virtual Solver<T>* doClone() const = 0;
};
}

namespace {
template <>
const char* const EnumStrings
    <N2D2::SGDSolver<float>::LearningRatePolicy>::data[]
    = {"None",
       "StepDecay",
       "ExponentialDecay",
       "InvTDecay",
       "PolyDecay",
       "InvDecay"};

template <>
const char* const EnumStrings
    <N2D2::SGDSolver<double>::LearningRatePolicy>::data[]
    = {"None",
       "StepDecay",
       "ExponentialDecay",
       "InvTDecay",
       "PolyDecay",
       "InvDecay"};
}

template <class T>
N2D2::SGDSolver<T>::SGDSolver()
    : mLearningRate(this, "LearningRate", 0.01),
      mMomentum(this, "Momentum", 0.0),
      mDecay(this, "Decay", 0.0),
      mPower(this, "Power", 0.0),
      mIterationSize(this, "IterationSize", 1U),
      mMaxIterations(this, "MaxIterations", 0U),
      mLearningRatePolicy(this, "LearningRatePolicy", None),
      mLearningRateStepSize(this, "LearningRateStepSize", 1U),
      mLearningRateDecay(this, "LearningRateDecay", 0.1),
      mClamping(this, "Clamping", false),
      mIterationPass(0),
      mNbIterations(0)
{
    // ctor
}

template <class T>
T N2D2::SGDSolver<T>::getLearningRate(unsigned int batchSize)
{
    if (mLearningRate == 0.0)
        return 0.0;

    if (mIterationPass < mIterationSize - 1) {
        ++mIterationPass;
        return 0.0;
    }
    else
        mIterationPass = 0;

    // Base learning rate
    T rate = mLearningRate;

    if (mLearningRatePolicy == SGDSolver<T>::StepDecay
        || mLearningRatePolicy == SGDSolver<T>::ExponentialDecay
        || mLearningRatePolicy == SGDSolver<T>::InvTDecay)
    {
        const unsigned int currentPattern = mNbIterations
                                            * mIterationSize * batchSize;
        const unsigned int currentStep = currentPattern / mLearningRateStepSize;

        if (mLearningRatePolicy == SGDSolver<T>::StepDecay)
            rate *= std::pow(mLearningRateDecay, (double)currentStep);
        else if (mLearningRatePolicy == SGDSolver<T>::ExponentialDecay)
            rate *= std::exp(-mLearningRateDecay * currentStep);
        else if (mLearningRatePolicy == SGDSolver<T>::InvTDecay)
            rate *= 1.0 / (1.0 + mLearningRateDecay * currentStep);

        if (mNbIterations > 0) {
            const unsigned int prevPattern = (mNbIterations - 1)
                                                * mIterationSize * batchSize;
            const unsigned int prevStep = prevPattern / mLearningRateStepSize;

            if (currentStep != prevStep) {
                std::cout << "Learning rate after " << mNbIterations
                          << "(x" << (mIterationSize * batchSize) << ") "
                          "iteration(s): " << rate << std::endl;
            }
        }
    }
    else if (mLearningRatePolicy == SGDSolver<T>::PolyDecay)
        rate *= std::pow(1.0 - (mNbIterations / (T)mMaxIterations), (T)mPower);
    else if (mLearningRatePolicy == SGDSolver<T>::InvDecay) {
        rate *= std::pow(1.0 + (mLearningRateDecay * mNbIterations),
                         -(T)mPower);
    }

    if (mMaxIterations == 0 || mNbIterations < mMaxIterations)
        ++mNbIterations;

    return rate;
}

#endif // N2D2_SGDSOLVER_H
