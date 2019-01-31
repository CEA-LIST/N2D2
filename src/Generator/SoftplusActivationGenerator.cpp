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

#include "Generator/SoftplusActivationGenerator.hpp"
#include "third_party/half.hpp"

N2D2::Registrar<N2D2::ActivationGenerator>
N2D2::SoftplusActivationGenerator::mRegistrar(
    "Softplus", N2D2::SoftplusActivationGenerator::generate);

std::shared_ptr<N2D2::SoftplusActivation>
N2D2::SoftplusActivationGenerator::generate(
    IniParser& /*iniConfig*/,
    const std::string& /*section*/,
    const std::string& model,
    const DataType& dataType,
    const std::string& /*name*/)
{
    return (dataType == Float32)
            ? Registrar<SoftplusActivation>::create<float>(model)()
        : (dataType == Float16)
            ? Registrar<SoftplusActivation>::create<half_float::half>(model)()
            : Registrar<SoftplusActivation>::create<double>(model)();
}
