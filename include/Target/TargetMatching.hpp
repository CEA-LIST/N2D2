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

#ifndef N2D2_TARGETMATCHING_H
#define N2D2_TARGETMATCHING_H

#include <map>
#include <string>
#include <vector>

#include "Monitor.hpp"
#include "Target.hpp"

namespace N2D2 {
class TargetMatching : public Target {
public:
    enum Distance {
        L1,
        L2,
        Linf
    };

    static std::shared_ptr<Target> create(const std::string& name,
                                          const std::shared_ptr<Cell>& cell,
                                          const std::shared_ptr
                                          <StimuliProvider>& sp,
                                          double targetValue = 1.0,
                                          double defaultValue = 0.0,
                                          unsigned int targetTopN = 1,
                                          const std::string& labelsMapping = "",
                                          bool createMissingLabels = false)
    {
        return std::make_shared<TargetMatching>(name,
                                             cell,
                                             sp,
                                             targetValue,
                                             defaultValue,
                                             targetTopN,
                                             labelsMapping,
                                             createMissingLabels);
    }
    static const char* Type;

    TargetMatching(const std::string& name,
                const std::shared_ptr<Cell>& cell,
                const std::shared_ptr<StimuliProvider>& sp,
                double targetValue = 1.0,
                double defaultValue = 0.0,
                unsigned int targetTopN = 1,
                const std::string& labelsMapping = "",
                bool createMissingLabels = false);
    virtual const char* getType() const
    {
        return Type;
    };
    bool newValidationEER(double validationEER);
    bool newValidationEER(double validationEER, double validationFRR);
    double getLastValidationEER() const
    {
        return (!mValidationEER.empty()) ? mValidationEER.back().second
                                         : 0.0;
    };
    double getMinValidationEER() const
    {
        return mMinValidationEER;
    };

    double getEER();
    double getFRR();
    double getFRR(double targetFAR);

    virtual void provideTargets(Database::StimuliSet /*set*/) {}
    virtual void process(Database::StimuliSet /*set*/);
    virtual void log(const std::string& fileName, Database::StimuliSet /*set*/);
    virtual void clear(Database::StimuliSet /*set*/);

protected:
    double computeDistance(const Tensor<Float_T>& a,
                           const Tensor<Float_T>& b) const;
    void computeDistances();
    std::pair<double, double> computeFAR_FRR(double threshold) const;

    /**
     * Compute the couple of values (FAR, FRR) in the ROC curve corresponding to
     * equation a * FAR + b * FRR + c = 0, with a precision of mROC_Precision.
    */
    std::pair<double, double> computeROC(double a, double b, double c);

    Parameter<Distance> mDistance;
    Parameter<unsigned int> mROC_NbSteps;
    Parameter<double> mROC_Precision;
    Parameter<double> mTargetFAR;

    unsigned int nbBatches;
    std::vector<std::pair<unsigned int, double> > mValidationEER;
    std::vector<std::pair<unsigned int, double> > mValidationFRR;
    double mMinValidationEER;
    double mMinValidationFRR;
    std::map<double, std::pair<double, double> > mROC;

    Tensor<int> mIDs;
    Tensor<Float_T> mSignatures;

    Float_T mDistanceMin;
    Float_T mDistanceMax;
    Tensor<Float_T> mDistances;

private:
    static Registrar<Target> mRegistrar;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::TargetMatching::Distance>::data[]
    = {"L1", "L2", "Linf"};
}

#endif // N2D2_TARGETMATCHING_H
