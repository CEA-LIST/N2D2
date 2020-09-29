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

#ifndef N2D2_TARGETAGGREGATE_H
#define N2D2_TARGETAGGREGATE_H

#include <map>
#include <string>
#include <vector>

#include "Target.hpp"
#include "utils/ConfusionMatrix.hpp"

namespace N2D2 {
class TargetAggregate : public Target {
public:
    struct Score {
        ConfusionMatrix<unsigned long long int> confusionMatrix;
        // Stimulus ID
        //       -> Target label ID
        //                -> Estimated label ID (vector): count
        std::map<unsigned int,
                 std::map<unsigned int,
                          std::vector<unsigned int> > > misclassified;
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
        return std::make_shared<TargetAggregate>(name,
                                            cell,
                                            sp,
                                            targetValue,
                                            defaultValue,
                                            targetTopN,
                                            labelsMapping,
                                            createMissingLabels);
    }
    static const char* Type;

    TargetAggregate(const std::string& name,
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
    unsigned int getNbTargets() const;
    void setROIsLabelTarget(const std::shared_ptr<Target>& target);
    void logConfusionMatrix(const std::string& fileName,
                            Database::StimuliSet set) const;
    void logMisclassified(const std::string& fileName,
                          Database::StimuliSet set) const;
    void clearConfusionMatrix(Database::StimuliSet set);
    void clearMisclassified(Database::StimuliSet set);
    virtual void process(Database::StimuliSet set);
    void processEstimatedLabels(Database::StimuliSet set,
                                Float_T* values = NULL);
    virtual void log(const std::string& fileName, Database::StimuliSet set);
    virtual void clear(Database::StimuliSet set);

protected:
    Parameter<unsigned int> mScoreTopN;

    std::map<Database::StimuliSet, Score> mScoreSet;
    std::shared_ptr<Target> mROIsLabelTarget;

private:
    static Registrar<Target> mRegistrar;
};
}

#endif // N2D2_TARGETAGGREGATE_H