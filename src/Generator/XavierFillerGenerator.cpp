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

#include "Generator/XavierFillerGenerator.hpp"

N2D2::Registrar<N2D2::FillerGenerator>
N2D2::XavierFillerGenerator::mRegistrar("XavierFiller",
                                        N2D2::XavierFillerGenerator::generate);

std::shared_ptr<N2D2::Filler>
N2D2::XavierFillerGenerator::generate(IniParser& iniConfig,
                                      const std::string& /*section*/,
                                      const std::string& name,
                                      const DataType& dataType)
{
    if (dataType == Float32) {
        const XavierFiller<float>::VarianceNorm varianceNorm
            = iniConfig.getProperty<XavierFiller<float>::VarianceNorm>(
                name + ".VarianceNorm", XavierFiller<float>::FanIn);
        const XavierFiller<float>::Distribution distribution
            = iniConfig.getProperty<XavierFiller<float>::Distribution>(
                name + ".Distribution", XavierFiller<float>::Uniform);
        const float scaling = iniConfig.getProperty<float>(name + ".Scaling",
                                                           1.0f);

        return std::make_shared<XavierFiller<float> >(varianceNorm,
                                                      distribution,
                                                      scaling);
    }
    else if (dataType == Float16) {
        const XavierFiller<half_float::half>::VarianceNorm varianceNorm
            = iniConfig.getProperty<XavierFiller<half_float::half>
                ::VarianceNorm>(name + ".VarianceNorm",
                                XavierFiller<half_float::half>::FanIn);
        const XavierFiller<half_float::half>::Distribution distribution
            = iniConfig.getProperty<XavierFiller<half_float::half>
                ::Distribution>(name + ".Distribution",
                                XavierFiller<half_float::half>::Uniform);
        const half_float::half scaling
            = iniConfig.getProperty<half_float::half>(name + ".Scaling",
                                                      half_float::half(1.0f));

        return std::make_shared<XavierFiller<half_float::half> >(varianceNorm,
                                                      distribution,
                                                      scaling);
    }
    else {
        const XavierFiller<double>::VarianceNorm varianceNorm
            = iniConfig.getProperty<XavierFiller<double>::VarianceNorm>(
                name + ".VarianceNorm", XavierFiller<double>::FanIn);
        const XavierFiller<double>::Distribution distribution
            = iniConfig.getProperty<XavierFiller<double>::Distribution>(
                name + ".Distribution", XavierFiller<double>::Uniform);
        const double scaling = iniConfig.getProperty<double>(name + ".Scaling",
                                                           1.0);

        return std::make_shared<XavierFiller<double> >(varianceNorm,
                                                       distribution,
                                                       scaling);
    }
}
