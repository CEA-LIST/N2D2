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

#ifndef N2D2_TARGETSCORE_H
#define N2D2_TARGETSCORE_H

#include <map>
#include <string>
#include <vector>

#include "Monitor.hpp"
#include "Target.hpp"
#include "utils/ConfusionMatrix.hpp"

namespace N2D2 {
class TargetScore : public Target {
public:
    struct Score {
        std::deque<double> success;
        ConfusionMatrix<unsigned long long int> confusionMatrix;
        std::vector<std::pair<unsigned int, unsigned int> > misclassified;
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
        return std::make_shared<TargetScore>(name,
                                             cell,
                                             sp,
                                             targetValue,
                                             defaultValue,
                                             targetTopN,
                                             labelsMapping);
    }
    static const char* Type;

    TargetScore(const std::string& name,
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
    bool newValidationScore(double validationScore);
    bool newValidationTopNScore(double validationTopNScore); // top-N accuracy
    double getLastValidationScore() const
    {
        return (!mValidationScore.empty()) ? mValidationScore.back().second
                                           : 0.0;
    };
    double getMaxValidationScore() const
    {
        return mMaxValidationScore;
    };
    double getLastTopNValidationScore() const
    {
        return (!mValidationTopNScore.empty())
                   ? mValidationTopNScore.back().second
                   : 0.0;
    };
    double getMaxTopNValidationScore() const
    {
        return mMaxValidationTopNScore;
    };
    const std::vector<double>& getBatchSuccess() const
    {
        return mBatchSuccess;
    }
    const std::vector<double>& getBatchTopNSuccess() const
    {
        return mBatchTopNSuccess;
    } // top-N accuracy
    double getBatchAverageSuccess() const;
    double getBatchAverageTopNSuccess() const; // top-N accuracy
    const Score& getScore(Database::StimuliSet set) const
    {
        return (*mScoreSet.find(set)).second;
    };
    const Score& getTopNScore(Database::StimuliSet set) const
    {
        return (*mScoreTopNSet.find(set)).second;
    }; // top-N accuracy
    double getAverageSuccess(Database::StimuliSet set,
                             unsigned int avgWindow = 0) const;
    double getAverageTopNSuccess(Database::StimuliSet set,
                                 unsigned int avgWindow
                                 = 0) const; // top-N accuracy
    void logSuccess(const std::string& fileName,
                    Database::StimuliSet set,
                    unsigned int avgWindow = 0) const;
    void logTopNSuccess(const std::string& fileName,
                        Database::StimuliSet set,
                        unsigned int avgWindow = 0) const; // top-N accuracy
    void logConfusionMatrix(const std::string& fileName,
                            Database::StimuliSet set) const;
    void logMisclassified(const std::string& fileName,
                          Database::StimuliSet set) const;
    void clearSuccess(Database::StimuliSet set);
    void clearConfusionMatrix(Database::StimuliSet set);
    void clearMisclassified(Database::StimuliSet set);
    void clearScore(Database::StimuliSet set);
    virtual void process(Database::StimuliSet set);
    virtual void log(const std::string& fileName, Database::StimuliSet set);
    virtual void clear(Database::StimuliSet set);

protected:
    std::vector<double> mBatchSuccess;
    std::vector<double> mBatchTopNSuccess; // top-N accuracy
    std::vector<std::pair<unsigned int, double> > mValidationScore;
    std::vector<std::pair<unsigned int, double> > mValidationTopNScore; // top-N
    // accuracy
    double mMaxValidationScore;
    double mMaxValidationTopNScore; // top-N accuracy
    std::map<Database::StimuliSet, Score> mScoreSet;
    std::map<Database::StimuliSet, Score> mScoreTopNSet; // top-N accuracy

private:
    static Registrar<Target> mRegistrar;
};
}

#endif // N2D2_TARGETSCORE_H
