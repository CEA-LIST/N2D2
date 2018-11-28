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

#include "Solver/SGDSolver.hpp"

const char* N2D2::SGDSolver::Type = "SGD";

N2D2::SGDSolver::SGDSolver()
    : mLearningRate(this, "LearningRate", 0.01),
      mMomentum(this, "Momentum", 0.0),
      mDecay(this, "Decay", 0.0),
      mPower(this, "Power", 0.0),
      mIterationSize(this, "IterationSize", 1U),
      mMaxIterations(this, "MaxIterations", 0U),
      mLearningRatePolicy(this, "LearningRatePolicy", None),
      mLearningRateStepSize(this, "LearningRateStepSize", 1U),
      mLearningRateDecay(this, "LearningRateDecay", 0.1),
      mQuantizationLevels(this, "QuantizationLevels", 0U),
      mClamping(this, "Clamping", false),
      mIterationPass(0),
      mNbIterations(0),
      mMinVal(0.0),
      mMaxVal(0.0),
      mMinValQuant(0.0),
      mMaxValQuant(0.0)
{
    // ctor
}

N2D2::SGDSolver::SGDSolver(const SGDSolver& solver)
    : mLearningRate(this, "LearningRate", solver.mLearningRate),
      mMomentum(this, "Momentum", solver.mMomentum),
      mDecay(this, "Decay", solver.mDecay),
      mPower(this, "Power", solver.mPower),
      mIterationSize(this, "IterationSize", solver.mIterationSize),
      mMaxIterations(this, "MaxIterations", solver.mMaxIterations),
      mLearningRatePolicy(this, "LearningRatePolicy",
                          solver.mLearningRatePolicy),
      mLearningRateStepSize(this, "LearningRateStepSize",
                            solver.mLearningRateStepSize),
      mLearningRateDecay(this, "LearningRateDecay", solver.mLearningRateDecay),
      mQuantizationLevels(this, "QuantizationLevels",
                          solver.mQuantizationLevels),
      mClamping(this, "Clamping", solver.mClamping),
      mIterationPass(solver.mIterationPass),
      mNbIterations(solver.mNbIterations),
      mMinVal(solver.mMinVal),
      mMaxVal(solver.mMaxVal),
      mMinValQuant(solver.mMinValQuant),
      mMaxValQuant(solver.mMaxValQuant)
{
    // copy-ctor
}

double N2D2::SGDSolver::getLearningRate(unsigned int batchSize)
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
    double rate = mLearningRate;

    if (mLearningRatePolicy == SGDSolver::StepDecay
        || mLearningRatePolicy == SGDSolver::ExponentialDecay
        || mLearningRatePolicy == SGDSolver::InvTDecay)
    {
        const unsigned int currentPattern = mNbIterations
                                            * mIterationSize * batchSize;
        const unsigned int currentStep = currentPattern / mLearningRateStepSize;

        if (mLearningRatePolicy == SGDSolver::StepDecay)
            rate *= std::pow(mLearningRateDecay, (double)currentStep);
        else if (mLearningRatePolicy == SGDSolver::ExponentialDecay)
            rate *= std::exp(-mLearningRateDecay * currentStep);
        else if (mLearningRatePolicy == SGDSolver::InvTDecay)
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
    else if (mLearningRatePolicy == SGDSolver::PolyDecay) {
        rate *= std::pow(1.0 - (mNbIterations / (double)mMaxIterations),
                         (double)mPower);
    }
    else if (mLearningRatePolicy == SGDSolver::InvDecay) {
        rate *= std::pow(1.0 + (mLearningRateDecay * mNbIterations),
                         -(double)mPower);
    }

    if (mMaxIterations == 0 || mNbIterations < mMaxIterations)
        ++mNbIterations;

    return rate;
}

void N2D2::SGDSolver::saveInternal(std::ostream& state,
                                   std::ostream& log) const
{
    state.write(reinterpret_cast<const char*>(&mMinVal), sizeof(mMinVal));
    state.write(reinterpret_cast<const char*>(&mMaxVal), sizeof(mMaxVal));
    state.write(reinterpret_cast<const char*>(&mMinValQuant),
                sizeof(mMinValQuant));
    state.write(reinterpret_cast<const char*>(&mMaxValQuant),
                sizeof(mMaxValQuant));

    log << "Range: [" << mMinVal << ", " << mMaxVal << "]\n"
        << "Quantization range (*Quant): [" << mMinValQuant << ", "
            << mMaxValQuant << "]" << std::endl;
}

void N2D2::SGDSolver::loadInternal(std::istream& state)
{
    state.read(reinterpret_cast<char*>(&mMinVal), sizeof(mMinVal));
    state.read(reinterpret_cast<char*>(&mMaxVal), sizeof(mMaxVal));
    state.read(reinterpret_cast<char*>(&mMinValQuant), sizeof(mMinValQuant));
    state.read(reinterpret_cast<char*>(&mMaxValQuant), sizeof(mMaxValQuant));
}
