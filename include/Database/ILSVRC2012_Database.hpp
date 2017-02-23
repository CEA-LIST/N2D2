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

#ifndef N2D2_ILSVRC2012_DATABASE_H
#define N2D2_ILSVRC2012_DATABASE_H

#include "Database/DIR_Database.hpp"

namespace N2D2 {
class ILSVRC2012_Database : public DIR_Database {
public:
    ILSVRC2012_Database(double learn, bool useValidationForTest = true);
    virtual void load(const std::string& dataPath,
                      const std::string& labelPath = "",
                      bool /*extractROIs*/ = false);
    virtual ~ILSVRC2012_Database() {};

protected:
    void loadImageNetValidationStimuli(const std::string& dirPath,
                                       const std::string& valNamePath);
    void loadImageNetStimuliPerDir(const std::string& dirPath,
                                   const std::string& labelNamePath);

    double mLearn;
    bool mUseValidationForTest;

};
}

#endif // N2D2_ILSVRC2012_DATABASE_H
