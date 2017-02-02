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
    virtual ~SGDSolver() {};

protected:
    /// Initial learning rate
    Parameter<double> mLearningRate;
    /// Momentum
    Parameter<double> mMomentum;
    /// Decay
    Parameter<double> mDecay;
    /// Power
    Parameter<double> mPower;
    /// MaxIterations
    Parameter<double> mMaxIterations;
    /// Learning rate decay policy
    Parameter<LearningRatePolicy> mLearningRatePolicy;
    /// Learning rate step size
    Parameter<unsigned int> mLearningRateStepSize;
    /// Learning rate decay
    Parameter<double> mLearningRateDecay;
    /// If true, don't clamp the weights between -1 and 1 during learning
    Parameter<bool> mClamping;

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
      mMaxIterations(this, "MaxIterations", 0.0),
      mLearningRatePolicy(this, "LearningRatePolicy", None),
      mLearningRateStepSize(this, "LearningRateStepSize", 1U),
      mLearningRateDecay(this, "LearningRateDecay", 0.1),
      mClamping(this, "Clamping", false),
      mNbIterations(0)
{
    // ctor
}

#endif // N2D2_SGDSOLVER_H
