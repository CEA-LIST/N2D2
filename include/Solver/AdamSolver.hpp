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

#ifndef N2D2_ADAMSOLVER_H
#define N2D2_ADAMSOLVER_H

#include "Solver/Solver.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {
class AdamSolver : public Solver {
public:
    typedef std::function<std::shared_ptr<AdamSolver>()> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static const char* Type;

    AdamSolver();
    const char* getType() const
    {
        return Type;
    };
    AdamSolver(const AdamSolver& solver);
    std::shared_ptr<AdamSolver> clone() const
    {
        return std::shared_ptr<AdamSolver>(doClone());
    }
    bool isNewIteration() const
    {
        return true;
    }
    std::pair<double, double> getRange() const
    {
        return std::make_pair(mMinVal, mMaxVal);
    }
    std::pair<double, double> getQuantizedRange() const
    {
        return std::make_pair(mMinValQuant, mMaxValQuant);
    }
    virtual ~AdamSolver() {};

protected:
    virtual void saveInternal(std::ostream& state, std::ostream& log) const;
    virtual void loadInternal(std::istream& state);

    /// Stepsize
    Parameter<double> mLearningRate;
    /// Exponential decay rate for the moment estimates
    Parameter<double> mBeta1;
    /// Exponential decay rate for the moment estimates
    Parameter<double> mBeta2;
    /// Epsilon
    Parameter<double> mEpsilon;
    /// Quantization levels (0 = no quantization)
    Parameter<unsigned int> mQuantizationLevels;
    /// If true, don't clamp the weights between -1 and 1 during learning
    Parameter<bool> mClamping;

    unsigned long long int mNbSteps;
    double mMinVal;
    double mMaxVal;
    double mMinValQuant;
    double mMaxValQuant;

private:
    virtual AdamSolver* doClone() const = 0;
};
}

#endif // N2D2_ADAMSOLVER_H
