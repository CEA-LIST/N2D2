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

#include "ComputerVision/LSL_Box.hpp"
#include "Target.hpp"
#include "utils/ConfusionMatrix.hpp"

namespace N2D2 {
class TargetBBox : public Target {
public:

    struct Score {
        ConfusionMatrix<unsigned long long int> confusionMatrix;
    };
    struct DetectedBBox {
        float x;
        float y;
        float w;
        float h;
        float cls;
        DetectedBBox() {}
        DetectedBBox(float x_, float y_,
                     float w_, float h_,
                     float cls_):
            x(x_), y(y_), w(w_), h(h_), cls(cls_) {}
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
        return std::make_shared<TargetBBox>(name,
                                          cell,
                                          sp,
                                          targetValue,
                                          defaultValue,
                                          targetTopN,
                                          labelsMapping);
    }
    static const char* Type;

    TargetBBox(const std::string& name,
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
    virtual void initialize();
    virtual void process(Database::StimuliSet set);

    cv::Mat drawEstimatedBBox(unsigned int batchPos = 0) const;

    virtual void logEstimatedLabels(const std::string& dirName) const;
    virtual void log(const std::string& fileName, Database::StimuliSet set);
    virtual void clear(Database::StimuliSet set);
    virtual ~TargetBBox();

protected:
    
    std::map<Database::StimuliSet, Score> mScoreSet;

    std::vector< std::vector<DetectedBBox> > mBatchDetectedBBox;

private:
    static Registrar<Target> mRegistrar;
};
}

#endif // N2D2_TARGETBBOX_H
