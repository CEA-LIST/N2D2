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

#ifndef N2D2_MNIST_IDX_DATABASE_H
#define N2D2_MNIST_IDX_DATABASE_H

#include "Database/IDX_Database.hpp"

namespace N2D2 {
class MNIST_IDX_Database : public IDX_Database {
public:
    MNIST_IDX_Database(double validation = 0.0);
    MNIST_IDX_Database(const std::string& dataPath,
                       const std::string& labelPath = "",
                       bool /*extractROIs*/ = false,
                       double validation = 0.0);

    virtual void load(const std::string& dataPath,
                      const std::string& labelPath = "",
                      bool /*extractROIs*/ = false);
    std::vector<unsigned int> loadRelationSample(Database::StimuliSet set);
    virtual ~MNIST_IDX_Database() {};

protected:
    double mValidation;

    std::map<unsigned int, std::vector<unsigned int>> mStimuliPerLabelTrain;
    std::map<unsigned int, std::vector<unsigned int>> mStimuliPerLabelTest;
};
}

#endif // N2D2_MNIST_IDX_DATABASE_H
