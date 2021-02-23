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
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"
#include "utils/Gnuplot.hpp"

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
        InvDecay,
        CosineDecay
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
    void logSchedule(const std::string& fileName,
                     unsigned int batchSize,
                     unsigned int epochSize = 0,
                     unsigned int maxSteps = 0);

    virtual ~SGDSolver() {};

protected:
    double getLearningRate(unsigned int batchSize, bool silent = false);
    template <class T> std::pair<T, T> getClamping() const;

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
    /// WarmUp Duration for Cosine Rules
    Parameter<unsigned int> mWarmUpDuration;
    //LR starting value for warmup = fraction of initial LR
    Parameter<double>  mWarmUpLRFrac;
    /// Learning rate decay policy
    Parameter<LearningRatePolicy> mLearningRatePolicy;
    /// Learning rate step size
    Parameter<unsigned int> mLearningRateStepSize;
    /// Learning rate decay
    Parameter<double> mLearningRateDecay;
    /// Weights clamping, format: "min:max", or ":max", or "min:", or empty
    Parameter<std::string> mClamping;
    // Polyak Momentum method use for param update: true by default 
    Parameter<bool> mPolyakMomentum;
    unsigned int mIterationPass;
    unsigned int mNbIterations;

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
       "InvDecay",
       "CosineDecay"};
}

template <class T>
std::pair<T, T> N2D2::SGDSolver::getClamping() const {
    T clampMin = std::numeric_limits<T>::lowest();
    T clampMax = std::numeric_limits<T>::max();

    if (!((std::string)mClamping).empty()) {
        const std::vector<std::string> clamping = Utils::split(mClamping, ":");

        if (clamping.size() != 2) {
#pragma omp critical
            throw std::runtime_error("SGDSolver::getClamping():"
                " wrong format for clamping");
        }

        if (!clamping[0].empty())
            clampMin = T(std::atof(clamping[0].c_str()));

        if (!clamping[1].empty())
            clampMax = T(std::atof(clamping[1].c_str()));
    }

    return std::make_pair(clampMin, clampMax);
}

#endif // N2D2_SGDSOLVER_H
