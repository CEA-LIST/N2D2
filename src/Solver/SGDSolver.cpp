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
      mWarmUpDuration(this, "WarmUpDuration", 0U),
      mLearningRatePolicy(this, "LearningRatePolicy", None),
      mLearningRateStepSize(this, "LearningRateStepSize", 1U),
      mLearningRateDecay(this, "LearningRateDecay", 0.1),
      mUniqueStep(this, "UniqueStep", 0U),
      mQuantizationLevels(this, "QuantizationLevels", 0U),
      mClamping(this, "Clamping", ""),
      mPolyakMomentum(this, "PolyakMomentum", true),
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
      mWarmUpDuration(this, "WarmUpDuration", solver.mWarmUpDuration),
      mLearningRatePolicy(this, "LearningRatePolicy",
                          solver.mLearningRatePolicy),
      mLearningRateStepSize(this, "LearningRateStepSize",
                            solver.mLearningRateStepSize),
      mLearningRateDecay(this, "LearningRateDecay", solver.mLearningRateDecay),
      mUniqueStep(this, "UniqueStep", solver.mUniqueStep),
      mQuantizationLevels(this, "QuantizationLevels",
                          solver.mQuantizationLevels),
      mClamping(this, "Clamping", solver.mClamping),
      mPolyakMomentum(this, "PolyakMomentum", solver.mPolyakMomentum),
      mIterationPass(solver.mIterationPass),
      mNbIterations(solver.mNbIterations),
      mMinVal(solver.mMinVal),
      mMaxVal(solver.mMaxVal),
      mMinValQuant(solver.mMinValQuant),
      mMaxValQuant(solver.mMaxValQuant)
{
    // copy-ctor
}

double N2D2::SGDSolver::getLearningRate(unsigned int batchSize, bool silent)
{
    if (mGlobalLearningRate > 0.0)
        return mGlobalLearningRate;

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
        || mLearningRatePolicy == SGDSolver::InvTDecay
        || mLearningRatePolicy == SGDSolver::CosineDecay)
    {
        if (!(mLearningRateStepSize > 0)) {
            throw std::runtime_error("SGDSolver::getLearningRate(): parameter"
                " mLearningRateStepSize must be > 0 for \"StepDecay\","
                " \"ExponentialDecay\" and \"InvTDecay\" mLearningRatePolicy");
        }

        const unsigned int currentPattern = mNbIterations
                                            * mIterationSize * batchSize;
        const unsigned int currentStep = currentPattern / mLearningRateStepSize;

        if (mLearningRatePolicy == SGDSolver::StepDecay)
            rate *= std::pow(mLearningRateDecay, (double)currentStep);
        else if (mLearningRatePolicy == SGDSolver::ExponentialDecay)
            rate *= std::exp(-mLearningRateDecay * currentStep);
        else if (mLearningRatePolicy == SGDSolver::InvTDecay)
            rate *= 1.0 / (1.0 + mLearningRateDecay * currentStep);
        else if (mLearningRatePolicy == SGDSolver::CosineDecay)
        {
            if(mWarmUpDuration > currentStep)
            {
                rate *= currentStep / (double) mWarmUpDuration;
            }
            else
            {
                const unsigned int step = currentStep - mWarmUpDuration;
                double cosine_decay = 0.5 * (1.0 
                                    + (double) std::cos(M_PI * (double) step / 
                                     (double) (mMaxIterations - mWarmUpDuration)));
                rate *=  std::max(0.0, cosine_decay) ;
            }
        }
        if (mNbIterations > 0) {
            const unsigned int prevPattern = (mNbIterations - 1)
                                                * mIterationSize * batchSize;
            const unsigned int prevStep = prevPattern / mLearningRateStepSize;

            if (currentStep != prevStep && !silent) {
                std::cout << "Learning rate after " << mNbIterations
                          << "(x" << (mIterationSize * batchSize) << ") "
                          "iteration(s): " << rate << std::endl;
            }
        }
    }
    else if (mLearningRatePolicy == SGDSolver::PolyDecay) {
        if (!(mMaxIterations > 0)) {
            throw std::runtime_error("SGDSolver::getLearningRate(): parameter"
                " mMaxIterations must be > 0 for \"PolyDecay\""
                " mLearningRatePolicy");
        }

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

void N2D2::SGDSolver::logSchedule(const std::string& fileName,
                                  unsigned int batchSize,
                                  unsigned int epochSize,
                                  unsigned int maxSteps)
{
    const unsigned int maxLogSteps = 10000;
    const unsigned int iterationPass = mIterationPass;
    const unsigned int nbIterations = mNbIterations;

    const std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty()) {
#pragma omp critical
        Utils::createDirectories(dirName);
    }

    std::ofstream log(fileName.c_str());

    if (!log.good()) {
#pragma omp critical
        throw std::runtime_error("Could not create scheduling log file: "
                                 + fileName);
    }

    mIterationPass = 0;
    mNbIterations = 0;

    if (maxSteps == 0 && batchSize > 0)
        maxSteps = std::ceil(mMaxSteps / (double)batchSize);

    if (maxSteps == 0)
        return;

    double prevLearningRate = 0.0;
    unsigned int nextLog = mLogSteps;
    const unsigned int minStride = std::max(1U, maxSteps / maxLogSteps);

    for (unsigned int step = 0, prevStep = 0; step < maxSteps; ++step) {
        const unsigned int i = step * batchSize;
        const double learningRate = getLearningRate(batchSize, true);
        const bool isLog = (i >= nextLog || step == maxSteps - 1);

        if (isNewIteration() && (learningRate != prevLearningRate
                                 || isLog))
        {
            const unsigned int epoch = (epochSize > 0)
                ? (i / epochSize) : 0;

            if (prevStep == 0 || step >= prevStep + minStride) {
                log << step
                    << " " << mNbIterations
                    << " " << epoch
                    << " " << learningRate
                    << " " << ((isLog) ? "1" : "0") << "\n";

                prevStep = step;

                if (isLog)
                    nextLog += mLogSteps;
            }

            prevLearningRate = learningRate;
        }
    }

    log.close();

    mIterationPass = iterationPass;
    mNbIterations = nbIterations;

    Gnuplot gnuplot(fileName + ".gnu");
    gnuplot.set("grid");
    gnuplot.setTitle("Learning rate schedule");

    std::stringstream xLabelStr;
    xLabelStr << "# steps (batch size: " << batchSize << ", "
        "iteration size: " << (batchSize * mIterationSize) << ")";

    gnuplot.setXlabel(xLabelStr.str());

    if (epochSize > 0) {
        gnuplot.set("x2label", "\"# epoch\" tc rgb \"blue\"");
        gnuplot.set("xtics nomirror");
        gnuplot.set("x2tics tc rgb \"blue\"");
    }

    gnuplot.setYlabel("Learning rate");

    std::stringstream plotStr;
    plotStr << "using 1:4 with steps notitle";

    if (epochSize > 0)
        plotStr << ", '' using 3:(NaN) axes x2y1 notitle";

    plotStr << ", '' using 1:($5==1?$4:1/0) with points pt 8 lc 7 title \"log step\"";

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName, plotStr.str());
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
