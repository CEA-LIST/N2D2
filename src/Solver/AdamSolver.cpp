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
      mClamping(this, "Clamping", ""),
      mNbSteps(0)
{
    // ctor
}

N2D2::AdamSolver::AdamSolver(const AdamSolver& solver)
    : mLearningRate(this, "LearningRate", solver.mLearningRate),
      mBeta1(this, "Beta1", solver.mBeta1),
      mBeta2(this, "Beta2", solver.mBeta2),
      mEpsilon(this, "Epsilon", solver.mEpsilon),
      mClamping(this, "Clamping", solver.mClamping),
      mNbSteps(solver.mNbSteps)
{
    // copy-ctor
}
