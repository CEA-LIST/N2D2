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

#include "Generator/AffineTransformationGenerator.hpp"

N2D2::Registrar<N2D2::TransformationGenerator>
N2D2::AffineTransformationGenerator::mRegistrar(
    "AffineTransformation", N2D2::AffineTransformationGenerator::generate);

std::shared_ptr<N2D2::AffineTransformation>
N2D2::AffineTransformationGenerator::generate(IniParser& iniConfig,
                                              const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const AffineTransformation::Operator firstOperator
        = iniConfig.getProperty
          <AffineTransformation::Operator>("FirstOperator");
    const std::string firstValue = iniConfig.getProperty
                                   <std::string>("FirstValue");
    const AffineTransformation::Operator secondOperator
        = iniConfig.getProperty<AffineTransformation::Operator>(
            "SecondOperator", AffineTransformation::Plus);
    const std::string secondValue = iniConfig.getProperty
                                    <std::string>("SecondValue", "");

    std::shared_ptr<AffineTransformation> trans = std::make_shared
        <AffineTransformation>(
            firstOperator, firstValue, secondOperator, secondValue);
    return trans;
}
