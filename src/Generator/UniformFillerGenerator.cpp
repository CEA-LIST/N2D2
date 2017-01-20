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

#include "Generator/UniformFillerGenerator.hpp"

N2D2::Registrar<N2D2::FillerGenerator> N2D2::UniformFillerGenerator::mRegistrar(
    "UniformFiller", N2D2::UniformFillerGenerator::generate);

std::shared_ptr<N2D2::UniformFiller<N2D2::Float_T> >
N2D2::UniformFillerGenerator::generate(IniParser& iniConfig,
                                       const std::string& /*section*/,
                                       const std::string& name)
{
    const double min = iniConfig.getProperty<double>(name + ".Min", 0.0);
    const double max = iniConfig.getProperty<double>(name + ".Max", 1.0);

    return std::make_shared<UniformFiller<Float_T> >(min, max);
}
