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

#ifndef N2D2_TARGETROIS_H
#define N2D2_TARGETROIS_H

#include <map>
#include <string>
#include <vector>

#include "ComputerVision/LSL_Box.hpp"
#include "Target.hpp"
#include "utils/ConfusionMatrix.hpp"

namespace N2D2 {
class TargetROIs : public Target {
public:
    struct Score {
        ConfusionMatrix<unsigned long long int> confusionMatrix;
    };

    struct DetectedBB {
        DetectedBB(const std::shared_ptr<ROI>& bb_,
                   double score_,
                   const std::shared_ptr<ROI>& roi_,
                   double matching_,
                   bool duplicate_)
            : bb(bb_),
              score(score_),
              roi(roi_),
              matching(matching_),
              duplicate(duplicate_)
        {
        }

        std::shared_ptr<ROI> bb;
        double score;
        std::shared_ptr<ROI> roi;
        double matching;
        bool duplicate;
    };

    static std::shared_ptr<Target> create(const std::string& name,
                                          const std::shared_ptr<Cell>& cell,
                                          const std::shared_ptr
                                          <StimuliProvider>& sp,
                                          double targetValue = 1.0,
                                          double defaultValue = 0.0,
                                          unsigned int targetTopN = 1,
                                          const std::string& labelsMapping = "")
    {
        return std::make_shared<TargetROIs>(name,
                                            cell,
                                            sp,
                                            targetValue,
                                            defaultValue,
                                            targetTopN,
                                            labelsMapping);
    }
    static const char* Type;

    TargetROIs(const std::string& name,
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
    unsigned int getNbTargets() const;
    void setROIsLabelTarget(const std::shared_ptr<Target>& target)
    {
        mROIsLabelTarget = target;
    };
    void logConfusionMatrix(const std::string& fileName,
                            Database::StimuliSet set) const;
    void clearConfusionMatrix(Database::StimuliSet set);
    virtual void process(Database::StimuliSet set);
    cv::Mat drawEstimatedLabels(unsigned int batchPos = 0) const;
    const std::vector<DetectedBB>& getDetectedBB(unsigned int batchPos
                                                 = 0) const
    {
        return mDetectedBB.at(batchPos);
    };
    cv::Mat getBBData(const DetectedBB& bb, unsigned int batchPos = 0) const;
    virtual void logEstimatedLabels(const std::string& dirName) const;
    virtual void log(const std::string& fileName, Database::StimuliSet set);
    virtual void clear(Database::StimuliSet set);

protected:
    Parameter<unsigned int> mMinSize;
    Parameter<double> mMinOverlap;
    Parameter<unsigned int> mFilterMinHeight;
    Parameter<unsigned int> mFilterMinWidth;
    Parameter<double> mFilterMinAspectRatio;
    Parameter<double> mFilterMaxAspectRatio;
    Parameter<unsigned int> mMergeMaxHDist;
    Parameter<unsigned int> mMergeMaxVDist;

    std::vector<std::vector<DetectedBB> > mDetectedBB;
    std::map<Database::StimuliSet, Score> mScoreSet;
    std::shared_ptr<Target> mROIsLabelTarget;

private:
    static bool scoreCompare(const DetectedBB& lhs, const DetectedBB& rhs)
    {
        return (lhs.score > rhs.score);
    }

    static Registrar<Target> mRegistrar;
};
}

#endif // N2D2_TARGETROIS_H
