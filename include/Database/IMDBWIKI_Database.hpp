

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

#ifndef N2D2_IMDBWIKI_DATABASE_H
#define N2D2_IMDBWIKI_DATABASE_H

#include "Database.hpp"
#include "Database/DIR_Database.hpp"
#include "N2D2.hpp"
#include "utils/Utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace N2D2 {
class IMDBWIKI_Database : public DIR_Database {
public:
    struct FaceParameters {
        std::string full_path;
        std::string name;
        double x0;
        double y0;
        double x1;
        double y1;
        double dob;
        double photo_taken;
        double gender;
    };

    IMDBWIKI_Database(bool WikiSet,
                      bool IMDBSet,
                      bool CropFrame,
                      double learn,
                      double validation);
    virtual void load(const std::string& dataPath,
                      const std::string& labelPath = "",
                      bool /*extractROIs*/ = false);
    virtual ~IMDBWIKI_Database() {};

protected:
    static const std::locale csvIMDBLocale;
    void loadStimuli(const std::string& dirPath, const std::string& labelPath);
    std::vector<FaceParameters> loadFaceParameters(const std::string
                                                   & path) const;
    bool mWiki;
    bool mIMDB;
    bool mCrop;
    double mLearn;
    double mValidation;
    unsigned int mNbCorruptedFrames;
};
}

#endif // N2D2_IMDBWIKI_DATABASE_H
