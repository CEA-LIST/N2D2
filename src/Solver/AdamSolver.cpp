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

#include "Solver/AdamSolver.hpp"

const char* N2D2::AdamSolver::Type = "Adam";

N2D2::AdamSolver::AdamSolver()
    : mLearningRate(this, "LearningRate", 0.001),
      mBeta1(this, "Beta1", 0.9),
      mBeta2(this, "Beta2", 0.999),
      mEpsilon(this, "Epsilon", 1.0e-8),
      mQuantizationLevels(this, "QuantizationLevels", 0U),
      mClamping(this, "Clamping", ""),
      mNbSteps(0),
      mMinVal(0.0),
      mMaxVal(0.0),
      mMinValQuant(0.0),
      mMaxValQuant(0.0)
{
    // ctor
}

N2D2::AdamSolver::AdamSolver(const AdamSolver& solver)
    : mLearningRate(this, "LearningRate", solver.mLearningRate),
      mBeta1(this, "Beta1", solver.mBeta1),
      mBeta2(this, "Beta2", solver.mBeta2),
      mEpsilon(this, "Epsilon", solver.mEpsilon),
      mQuantizationLevels(this, "QuantizationLevels",
                          solver.mQuantizationLevels),
      mClamping(this, "Clamping", solver.mClamping),
      mNbSteps(solver.mNbSteps),
      mMinVal(solver.mMinVal),
      mMaxVal(solver.mMaxVal),
      mMinValQuant(solver.mMinValQuant),
      mMaxValQuant(solver.mMaxValQuant)
{
    // copy-ctor
}

void N2D2::AdamSolver::saveInternal(std::ostream& state,
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

void N2D2::AdamSolver::loadInternal(std::istream& state)
{
    state.read(reinterpret_cast<char*>(&mMinVal), sizeof(mMinVal));
    state.read(reinterpret_cast<char*>(&mMaxVal), sizeof(mMaxVal));
    state.read(reinterpret_cast<char*>(&mMinValQuant), sizeof(mMinValQuant));
    state.read(reinterpret_cast<char*>(&mMaxValQuant), sizeof(mMaxValQuant));
}
