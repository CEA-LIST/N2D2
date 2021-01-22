/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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

#ifndef N2D2_QUANTIZERACTIVATIONGENERATOR_H
#define N2D2_QUANTIZERACTIVATIONGENERATOR_H

#include "Quantizer/Activation/QuantizerActivation.hpp"
#include "utils/IniParser.hpp"
#include "utils/Registrar.hpp"
#include "Generator/Generator.hpp"

namespace N2D2 {
class QuantizerActivationGenerator {
public:
    typedef std::function
        <std::shared_ptr<QuantizerActivation>(IniParser& iniConfig,
                                           const std::string& section,
                                           const std::string& model,
                                           const DataType& dataType,
                                           const std::string& name)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static std::shared_ptr<QuantizerActivation>
    generate(IniParser& iniConfig,
             const std::string& section,
             const std::string& model,
             const DataType& dataType,
             const std::string& name,
             const std::shared_ptr<QuantizerActivation>& defaultQuantizer
             = std::shared_ptr<QuantizerActivation>());
};
}

#endif // N2D2_QUANTIZERACTIVATIONGENERATOR_H
