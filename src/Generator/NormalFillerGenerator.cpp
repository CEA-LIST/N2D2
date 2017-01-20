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

#include "Generator/NormalFillerGenerator.hpp"

N2D2::Registrar<N2D2::FillerGenerator>
N2D2::NormalFillerGenerator::mRegistrar("NormalFiller",
                                        N2D2::NormalFillerGenerator::generate);

std::shared_ptr<N2D2::NormalFiller<N2D2::Float_T> >
N2D2::NormalFillerGenerator::generate(IniParser& iniConfig,
                                      const std::string& /*section*/,
                                      const std::string& name)
{
    const double mean = iniConfig.getProperty<double>(name + ".Mean", 0.0);
    const double stdDev = iniConfig.getProperty<double>(name + ".StdDev", 1.0);

    return std::make_shared<NormalFiller<Float_T> >(mean, stdDev);
}
