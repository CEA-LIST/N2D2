/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
                    Vincent TEMPLIER (vincent.templier@cea.fr)

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

#include "Generator/QuantizerGenerator.hpp"

std::shared_ptr<N2D2::Quantizer>
N2D2::QuantizerGenerator::generate(IniParser& iniConfig,
                                const std::string& section,
                                const std::string& model,
                                const DataType& dataType,
                                const std::string& name,
                                const std::shared_ptr
                                <Quantizer>& defaultQuantizer)
{
    if (!iniConfig.currentSection(section, false))
        throw std::runtime_error("Missing [" + section + "] section.");

    if (iniConfig.isProperty(name)) {
        const std::string type = iniConfig.getProperty<std::string>(name);
        
        if (type.compare("NoQuant") == 0)
            return defaultQuantizer;

        return Registrar<QuantizerGenerator>::create(type)(iniConfig,
                                                        section,
                                                        model,
                                                        dataType,
                                                        name);
    } else
        return defaultQuantizer;
}
