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

#include "Generator/RandomResizeCropTransformationGenerator.hpp"

N2D2::Registrar<N2D2::TransformationGenerator>
N2D2::RandomResizeCropTransformationGenerator::mRegistrar(
    "RandomResizeCropTransformation",
    N2D2::RandomResizeCropTransformationGenerator::generate);

std::shared_ptr<N2D2::RandomResizeCropTransformation>
N2D2::RandomResizeCropTransformationGenerator::generate(IniParser& iniConfig,
                                                       const std::string
                                                       & section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const unsigned int width = iniConfig.getProperty<unsigned int>("Width");
    const unsigned int height = iniConfig.getProperty<unsigned int>("Height");

    std::shared_ptr<RandomResizeCropTransformation> trans = std::make_shared
        <RandomResizeCropTransformation>(width, height);
    trans->setParameters(iniConfig.getSection(section, true));
    return trans;
}
