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

#ifndef N2D2_TARGETCOMPARE_H
#define N2D2_TARGETCOMPARE_H

#include <map>
#include <string>
#include <vector>

#include "Target.hpp"
#include "TargetScore.hpp"

namespace N2D2 {
class TargetCompare : public TargetScore {
public:
    static std::shared_ptr<Target> create(
        const std::string& name,
        const std::shared_ptr<Cell>& cell,
        const std::shared_ptr<StimuliProvider>& sp,
        double targetValue = 1.0,
        double defaultValue = 0.0,
        unsigned int targetTopN = 1,
        const std::string& labelsMapping = "")
    {
        return std::make_shared<TargetCompare>(name,
                                          cell,
                                          sp,
                                          targetValue,
                                          defaultValue,
                                          targetTopN,
                                          labelsMapping);
    }
    static const char* Type;

    TargetCompare(const std::string& name,
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
    virtual void process(Database::StimuliSet set);
    virtual ~TargetCompare();

protected:
    Parameter<std::string> mDataPath;
    Parameter<std::string> mMatching;

private:
    static Registrar<Target> mRegistrar;
};
}

#endif // N2D2_TARGETCOMPARE_H
