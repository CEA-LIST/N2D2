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

#ifndef N2D2_TARGETRP_H
#define N2D2_TARGETRP_H

#include <map>
#include <string>
#include <vector>

#include "ComputerVision/LSL_Box.hpp"
#include "Target.hpp"
#include "Cell/RPCell.hpp"
#include "Cell/AnchorCell.hpp"
#include "utils/ConfusionMatrix.hpp"

namespace N2D2 {
class TargetRP : public Target {
public:
    enum TargetType {
        Cls,
        BBox
    };

    struct Score {
        ConfusionMatrix<unsigned long long int> confusionMatrix;
    };

    static std::shared_ptr<Target> create(
        const std::string& name,
        const std::shared_ptr<Cell>& cell,
        const std::shared_ptr<StimuliProvider>& sp,
        double targetValue = 1.0,
        double defaultValue = 0.0,
        unsigned int targetTopN = 1,
        const std::string& labelsMapping = "")
    {
        return std::make_shared<TargetRP>(name,
                                          cell,
                                          sp,
                                          targetValue,
                                          defaultValue,
                                          targetTopN,
                                          labelsMapping);
    }
    static const char* Type;

    TargetRP(const std::string& name,
             const std::shared_ptr<Cell>& cell,
             const std::shared_ptr<StimuliProvider>& sp,
             double targetValue = 1.0,
             double defaultValue = 0.0,
             unsigned int targetTopN = 1,
             const std::string& labelsMapping = "");
    virtual const char* getType() const
    {
        return Type;
    };
    const Score& getScore(Database::StimuliSet set) const
    {
        return (*mScoreSet.find(set)).second;
    };
    void logConfusionMatrix(const std::string& fileName,
                            Database::StimuliSet set) const;
    void clearConfusionMatrix(Database::StimuliSet set);
    virtual void initialize(TargetType targetType,
                            const std::shared_ptr<RPCell>& RPCell,
                            const std::shared_ptr<AnchorCell>& anchorCell);
    virtual void process(Database::StimuliSet set);
    virtual void processCls(Database::StimuliSet set);
    virtual void processBBox(Database::StimuliSet set);
    cv::Mat drawEstimatedLabels(unsigned int batchPos = 0) const;
    virtual void logEstimatedLabels(const std::string& dirName) const;
    virtual void log(const std::string& fileName, Database::StimuliSet set);
    virtual void clear(Database::StimuliSet set);
    virtual ~TargetRP();

protected:
    inline Float_T smoothL1(Float_T tx, Float_T x) const;

    Parameter<double> mMinOverlap;
    Parameter<double> mLossLambda;

    TargetType mTargetType;
    std::shared_ptr<RPCell> mRPCell;
    std::shared_ptr<AnchorCell> mAnchorCell;
    std::map<Database::StimuliSet, Score> mScoreSet;

    static std::map<std::string, std::map<TargetType, TargetRP*> > mTargetRP;

private:
    static Registrar<Target> mRegistrar;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::TargetRP::TargetType>::data[]
    = {"Cls", "BBox"};
}

N2D2::Float_T N2D2::TargetRP::smoothL1(Float_T tx, Float_T x) const {
    const Float_T error = tx - x;
    const Float_T sign = (error >= 0.0) ? 1.0 : -1.0;
    return (std::fabs(error) < 1.0) ? error : sign;
}

#endif // N2D2_TARGETRP_H
