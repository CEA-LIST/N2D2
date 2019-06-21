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

#ifndef N2D2_ACTITRACKER_DATABASE_H
#define N2D2_ACTITRACKER_DATABASE_H

#include "Database/Database.hpp"

namespace N2D2 {
class Actitracker_Database : public Database {
public:
    struct RawData {
        std::string user;
        std::string activity;
        unsigned long long int timestamp;
        float xAcceleration;
        float yAcceleration;
        float zAcceleration;
    };

    Actitracker_Database(double learn = 0.6, double validation = 0.2,
                         bool useUnlabeledForTest = false);
    virtual void load(const std::string& dataPath,
                      const std::string& labelPath = "",
                      bool /*extractROIs*/ = false);
    void loadRaw(const std::string& fileName);
    virtual ~Actitracker_Database() {};

protected:
    Parameter<unsigned int> mWindowSize;
    Parameter<double> mOverlapping;

    double mLearn;
    double mValidation;
    bool mUseUnlabeledForTest;

    std::locale mCsvLocale;
};
}

#endif // N2D2_ACTITRACKER_DATABASE_H
