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

#ifndef N2D2_MAPPINGGENERATOR_H
#define N2D2_MAPPINGGENERATOR_H

#include <memory>
#include <string>

#include "Cell/Cell.hpp"
#include "StimuliProvider.hpp"
#include "utils/IniParser.hpp"

namespace N2D2 {
class MappingGenerator {
public:
    struct Mapping {
        unsigned int sizeX;
        unsigned int sizeY;
        unsigned int strideX;
        unsigned int strideY;
        unsigned int offsetX;
        unsigned int offsetY;
        unsigned int nbIterations;
    };

    static const Mapping defaultMapping;
    static Mapping getMapping(IniParser& iniConfig,
                              const std::string& section,
                              const std::string& name,
                              const Mapping& defaultMapping_ = defaultMapping);

    static Matrix<bool> generate(StimuliProvider& sp,
                                 std::shared_ptr<Cell> parent,
                                 unsigned int nbChannels,
                                 IniParser& iniConfig,
                                 const std::string& section,
                                 const Mapping& defaultMapping_
                                 = defaultMapping);
};
}

#endif // N2D2_MAPPINGGENERATOR_H
