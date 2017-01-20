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

#include "Generator/FilterTransformationGenerator.hpp"

N2D2::Registrar<N2D2::TransformationGenerator>
N2D2::FilterTransformationGenerator::mRegistrar(
    "FilterTransformation", N2D2::FilterTransformationGenerator::generate);

std::shared_ptr<N2D2::FilterTransformation>
N2D2::FilterTransformationGenerator::generate(IniParser& iniConfig,
                                              const std::string& section)
{
    if (!iniConfig.currentSection(section, false))
        throw std::runtime_error("Missing [" + section + "] section.");

    const std::string kernelName = iniConfig.getProperty<std::string>("Kernel");

    double orientation = 0.0;

    if (iniConfig.isProperty("Orientation"))
        orientation = iniConfig.getProperty<double>("Orientation");
    else if (kernelName == "Gabor") {
        const double theta
            = Utils::degToRad(iniConfig.getProperty<double>("Kernel.Theta"));
        orientation = Utils::normalizedAngle(theta, Utils::ZeroToTwoPi)
                      / (2.0 * M_PI);
    }

    const Kernel<double> kernel = KernelGenerator::generate
        <double>(iniConfig, section, "Kernel");

    return std::make_shared<FilterTransformation>(kernel, orientation);
}
