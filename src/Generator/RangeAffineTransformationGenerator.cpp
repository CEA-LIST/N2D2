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

#include "Generator/RangeAffineTransformationGenerator.hpp"

N2D2::Registrar<N2D2::TransformationGenerator>
N2D2::RangeAffineTransformationGenerator::mRegistrar(
    "RangeAffineTransformation",
    N2D2::RangeAffineTransformationGenerator::generate);

std::shared_ptr<N2D2::RangeAffineTransformation>
N2D2::RangeAffineTransformationGenerator::generate(IniParser& iniConfig,
                                                   const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const RangeAffineTransformation::Operator firstOperator
        = iniConfig.getProperty
          <RangeAffineTransformation::Operator>("FirstOperator");
    const double firstValue = iniConfig.getProperty<double>("FirstValue");
    const RangeAffineTransformation::Operator secondOperator
        = iniConfig.getProperty<RangeAffineTransformation::Operator>(
            "SecondOperator", RangeAffineTransformation::Plus);
    const double secondValue = iniConfig.getProperty
                               <double>("SecondValue", 0.0);

    std::shared_ptr<RangeAffineTransformation> trans = std::make_shared
        <RangeAffineTransformation>(
            firstOperator, firstValue, secondOperator, secondValue);
    return trans;
}
