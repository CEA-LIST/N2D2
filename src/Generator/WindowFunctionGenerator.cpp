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

#include "Generator/WindowFunctionGenerator.hpp"

std::string N2D2::WindowFunctionGenerator::defaultWindow = "Rectangular";
double N2D2::WindowFunctionGenerator::defaultSigma = 0.4;
double N2D2::WindowFunctionGenerator::defaultAlpha = 0.16;
double N2D2::WindowFunctionGenerator::defaultBeta = 5.0;

void N2D2::WindowFunctionGenerator::setDefault(IniParser& iniConfig,
                                               const std::string& section,
                                               const std::string& name)
{
    if (!iniConfig.currentSection(section, false))
        throw std::runtime_error("Missing [" + section + "] section.");

    defaultWindow = iniConfig.getProperty<std::string>(name, defaultWindow);
    defaultSigma = iniConfig.getProperty<double>(name + ".Sigma", defaultSigma);
    defaultAlpha = iniConfig.getProperty<double>(name + ".Alpha", defaultAlpha);
    defaultBeta = iniConfig.getProperty<double>(name + ".Beta", defaultBeta);
}
