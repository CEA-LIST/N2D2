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

#include "Generator/RandomAffineTransformationGenerator.hpp"

N2D2::Registrar<N2D2::TransformationGenerator>
N2D2::RandomAffineTransformationGenerator::mRegistrar(
    "RandomAffineTransformation",
    N2D2::RandomAffineTransformationGenerator::generate);

std::shared_ptr<N2D2::RandomAffineTransformation>
N2D2::RandomAffineTransformationGenerator::generate(IniParser& iniConfig,
                                                    const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const std::vector<std::pair<double, double> > gainRange
        = getRanges(iniConfig, section, "GainRange", std::make_pair(1.0, 1.0));
    const std::vector<std::pair<double, double> > biasRange
        = getRanges(iniConfig, section, "BiasRange", std::make_pair(0.0, 0.0));
    const std::vector<std::pair<double, double> > gammaRange
        = getRanges(iniConfig, section, "GammaRange", std::make_pair(1.0, 1.0));

    const std::vector<double> gainVarProb
        = iniConfig.getProperty<std::vector<double> >("GainVarProb",
                                                      std::vector<double>());
    const std::vector<double> biasVarProb
        = iniConfig.getProperty<std::vector<double> >("BiasVarProb",
                                                      std::vector<double>());
    const std::vector<double> gammaVarProb
        = iniConfig.getProperty<std::vector<double> >("GammaVarProb",
                                                      std::vector<double>());

    std::shared_ptr<RandomAffineTransformation> trans = std::make_shared
        <RandomAffineTransformation>(gainRange, biasRange, gammaRange,
                                     gainVarProb, biasVarProb, gammaVarProb);
    trans->setParameters(iniConfig.getSection(section, true));
    return trans;
}

std::pair<double, double>
N2D2::RandomAffineTransformationGenerator::getRange(IniParser& iniConfig,
                                                    const std::string& section,
                                                    const std::string& property)
{
    std::vector<double> range
        = iniConfig.getProperty<std::vector<double> >(property);

    if (range.size() != 2) {
        throw std::runtime_error(property + " must have two values "
                                 "(\"min max\") in section " + section);
    }

    return std::make_pair(range[0], range[1]);
}

std::vector<std::pair<double, double> >
N2D2::RandomAffineTransformationGenerator::getRanges(
    IniParser& iniConfig,
    const std::string& section,
    const std::string& property,
    std::pair<double, double> defaultRange)
{
    std::vector<std::pair<double, double> > ranges;

    if (iniConfig.isProperty(property + "[*]")) {
        ranges.resize(3);

        for (unsigned int ch = 0; ch < 3; ++ch) {
            std::stringstream propertyStr;
            propertyStr << property << "[" << ch << "]";

            if (iniConfig.isProperty(propertyStr.str()))
                ranges[ch] = getRange(iniConfig, section, propertyStr.str());
            else
                ranges[ch] = defaultRange;
        }
    }
    else if (iniConfig.isProperty(property))
        ranges.push_back(getRange(iniConfig, section, property));

    return ranges;
}
