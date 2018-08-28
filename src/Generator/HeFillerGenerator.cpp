/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Generator/HeFillerGenerator.hpp"

N2D2::Registrar<N2D2::FillerGenerator>
N2D2::HeFillerGenerator::mRegistrar("HeFiller",
                                    N2D2::HeFillerGenerator::generate);

std::shared_ptr<N2D2::Filler>
N2D2::HeFillerGenerator::generate(IniParser& iniConfig,
                                  const std::string& /*section*/,
                                  const std::string& name,
                                  const DataType& dataType)
{
    if (dataType == Float32) {
        const HeFiller<float>::VarianceNorm varianceNorm
            = iniConfig.getProperty<HeFiller<float>::VarianceNorm>(
                name + ".VarianceNorm", HeFiller<float>::FanIn);

        return std::make_shared<HeFiller<float> >(varianceNorm);
    }
    else if (dataType == Float16) {
        const HeFiller<half_float::half>::VarianceNorm varianceNorm
            = iniConfig.getProperty<HeFiller<half_float::half>
                ::VarianceNorm>(name + ".VarianceNorm",
                                HeFiller<half_float::half>::FanIn);

        return std::make_shared<HeFiller<half_float::half> >(varianceNorm);
    }
    else {
        const HeFiller<double>::VarianceNorm varianceNorm
            = iniConfig.getProperty<HeFiller<double>::VarianceNorm>(
                name + ".VarianceNorm", HeFiller<double>::FanIn);

        return std::make_shared<HeFiller<double> >(varianceNorm);
    }
}
