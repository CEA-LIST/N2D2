/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_CITYSCAPES_DATABASE_H
#define N2D2_CITYSCAPES_DATABASE_H

#include <jsoncpp/json/json.h>

#include "Database.hpp"
#include "Database/DIR_Database.hpp"
#include "N2D2.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace N2D2 {
class Cityscapes_Database : public DIR_Database {
public:
    struct Color {
        Color(unsigned char red_, unsigned char green_, unsigned char blue_):
            red(red_), green(green_), blue(blue_) {}

        unsigned char red;
        unsigned char green;
        unsigned char blue;
    };

    struct Label {
        Label(const std::string& name_,
              int id_,
              const std::string& category_,
              int categoryId_,
              bool hasInstances_,
              bool ignoreEval_,
              const Color& color_):
            name(name_),
            id(id_),
            category(category_),
            categoryId(categoryId_),
            hasInstances(hasInstances_),
            ignoreEval(ignoreEval_),
            color(color_) {}

        std::string name;
        int id;
        std::string category;
        int categoryId;
        bool hasInstances;
        bool ignoreEval;
        Color color;
    };

    Cityscapes_Database(bool incTrainExtra = false,
                        bool useCoarse = false,
                        bool singleInstanceLabels = true);
    virtual void load(const std::string& dataPath,
                      const std::string& labelPath = "",
                      bool /*extractROIs*/ = false);
    virtual ~Cityscapes_Database() {};

protected:
    void loadLabels(const std::string& labelPath);

    bool mIncTrainExtra;
    bool mUseCoarse;
    bool mSingleInstanceLabels;
    std::vector<Label> mLabels;
};
}

#endif // N2D2_CITYSCAPES_DATABASE_H
