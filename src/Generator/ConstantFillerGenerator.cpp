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

#include "Generator/ConstantFillerGenerator.hpp"

N2D2::Registrar<N2D2::FillerGenerator>
N2D2::ConstantFillerGenerator::mRegistrar(
    "ConstantFiller", N2D2::ConstantFillerGenerator::generate);

std::shared_ptr<N2D2::Filler>
N2D2::ConstantFillerGenerator::generate(IniParser& iniConfig,
                                        const std::string& /*section*/,
                                        const std::string& name,
                                        const DataType& dataType)
{
    if (dataType == Float32) {
        const float value = iniConfig.getProperty<float>(name + ".Value");

        return std::make_shared<ConstantFiller<float> >(value);
    }
    else if (dataType == Float16) {
        const half_float::half value
            = iniConfig.getProperty<half_float::half>(name + ".Value");

        return std::make_shared<ConstantFiller<half_float::half> >(value);
    }
    else {
        const double value = iniConfig.getProperty<double>(name + ".Value");

        return std::make_shared<ConstantFiller<double> >(value);
    }
}
