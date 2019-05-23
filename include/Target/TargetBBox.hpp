/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
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

#ifndef N2D2_TARGETBBOX_H
#define N2D2_TARGETBBOX_H

#include <map>
#include <string>
#include <vector>

#include "Monitor.hpp"
#include "ComputerVision/LSL_Box.hpp"
#include "Target.hpp"
#include "utils/ConfusionMatrix.hpp"

namespace N2D2 {
class TargetBBox : public Target {
public:

    struct ScoreBBox {
        ConfusionMatrix<unsigned long long int> confusionMatrix;
        std::vector<std::deque<double>> success;
    };

    struct DetectedBBox {
        float x;
        float y;
        float w;
        float h;
        float conf;
        int cls;
        DetectedBBox() {}
        DetectedBBox(float x_, float y_,
                     float w_, float h_,
                     float conf_, int cls_):
            x(x_), y(y_), w(w_), h(h_), conf(conf_), cls(cls_) {}
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

    static std::shared_ptr<Target> create(
        const std::string& name,
        const std::shared_ptr<Cell>& cell,
        const std::shared_ptr<StimuliProvider>& sp,
        double targetValue = 1.0,
        double defaultValue = 0.0,
        unsigned int targetTopN = 1,
        const std::string& labelsMapping = "",
        bool createMissingLabels = false)
    {
        return std::make_shared<TargetBBox>(name,
                                          cell,
                                          sp,
                                          targetValue,
                                          defaultValue,
                                          targetTopN,
                                          labelsMapping,
                                          createMissingLabels);
    }
    static const char* Type;

    TargetBBox(const std::string& name,
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
    const ScoreBBox& getScore(Database::StimuliSet set) const
    {
        return (*mScoreSet.find(set)).second;
    };
    double getMaxValidationScore() const
    {
        return mMaxValidationScore;
    };
    double getLastValidationScore() const
    {
        return (!mValidationScore.empty()) ? mValidationScore.back().second
                                           : 0.0;
    };
    double getBBoxSucess(Database::StimuliSet set);
    void logSuccess(const std::string& fileName,
                    Database::StimuliSet set,
                    unsigned int avgWindow = 0) const;

    void logConfusionMatrix(const std::string& fileName,
                            Database::StimuliSet set) const;
    void clearConfusionMatrix(Database::StimuliSet set);
    bool newValidationScore(double validationScore);

    virtual void initialize(bool genAnchors = false, 
                            unsigned int nbAnchors = 6U, 
                            long unsigned int nbIterMax = 100000);

    virtual void process(Database::StimuliSet set);
    double getAverageSuccess(Database::StimuliSet set,
                             unsigned int avgWindow = 0) const;

    cv::Mat drawEstimatedBBox(unsigned int batchPos = 0) const;
    const std::vector<DetectedBB>& getDetectedBB(unsigned int batchPos
                                                 = 0) const
    {
        return mBatchDetectedBBox.at(batchPos);
    };

    virtual void logEstimatedLabels(const std::string& dirName) const;
    virtual void log(const std::string& fileName, Database::StimuliSet set);
    virtual void clear(Database::StimuliSet set);
    void clearSuccess(Database::StimuliSet set);

    virtual ~TargetBBox();

protected:
    std::vector<std::vector<DetectedBB> > mBatchDetectedBBox;
    std::vector<std::pair<unsigned int, double> > mValidationScore;

    std::map<Database::StimuliSet, ScoreBBox> mScoreSet;
    std::vector<std::string> mLabelsBBoxName;

    //std::vector< std::vector<DetectedBBox> > mBatchDetectedBBox;

    bool mGenerateAnchors;
    unsigned int mNbAnchors;
    long unsigned int mIterMax;
    double mMaxValidationScore = 0.0;

private:
    static bool scoreCompare(const DetectedBB& lhs, const DetectedBB& rhs)
    {
        return (lhs.score > rhs.score);
    }

    static Registrar<Target> mRegistrar;
};
}

#endif // N2D2_TARGETBBOX_H
