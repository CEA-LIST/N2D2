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

#include "Generator/ActivationCellGenerator.hpp"
#include "DeepNet.hpp"
#include "StimuliProvider.hpp"

N2D2::Registrar<N2D2::CellGenerator> N2D2::ActivationCellGenerator::mRegistrar(
    ActivationCell::Type, N2D2::ActivationCellGenerator::generate);

std::shared_ptr<N2D2::ActivationCell>
N2D2::ActivationCellGenerator::generate(Network& /*network*/, const DeepNet& deepNet,
                                       StimuliProvider& sp,
                                       const std::vector
                                       <std::shared_ptr<Cell> >& parents,
                                       IniParser& iniConfig,
                                       const std::string& section)
{
    if (!iniConfig.currentSection(section, false))
        throw std::runtime_error("Missing [" + section + "] section.");

    const std::string model = iniConfig.getProperty<std::string>(
        "Model", CellGenerator::mDefaultModel);
    const DataType dataType = iniConfig.getProperty<DataType>(
        "DataType", CellGenerator::mDefaultDataType);

    const unsigned int nbOutputs = iniConfig.getProperty
                                   <unsigned int>("NbOutputs");

    std::cout << "Layer: " << section << " [Activation(" << model << ")]"
              << std::endl;

    std::shared_ptr<Activation> activation
        = ActivationGenerator::generate(
            iniConfig,
            section,
            model,
            dataType,
            "ActivationFunction",
            (dataType == Float32)
                ? Registrar<TanhActivation>::create<float>(model)()
            : (dataType == Float16)
                ? Registrar<TanhActivation>::create<half_float::half>(model)()
                : Registrar<TanhActivation>::create<double>(model)());

    // Cell construction
    std::shared_ptr<ActivationCell> cell
        = (dataType == Float32)
            ? Registrar<ActivationCell>::create<float>(model)(deepNet, section,
                                                             nbOutputs,
                                                             activation)
          : (dataType == Float16)
            ? Registrar<ActivationCell>::create<half_float::half>(model)(deepNet, section,
                                                                 nbOutputs,
                                                                 activation)
            : Registrar<ActivationCell>::create<double>(model)(deepNet, section,
                                                              nbOutputs,
                                                              activation);

    if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in network configuration file: " + iniConfig.getFileName());
    }

    // Connect the cell to the parents
    for (std::vector<std::shared_ptr<Cell> >::const_iterator it
         = parents.begin(),
         itEnd = parents.end();
         it != itEnd;
         ++it) {
        if (!(*it))
            cell->addInput(sp, 0, 0, sp.getSizeX(), sp.getSizeY());
        else
            cell->addInput((*it).get());
    }

    std::cout << "  # Inputs dims: " << cell->getInputsDims() << std::endl;
    std::cout << "  # Outputs dims: " << cell->getOutputsDims() << std::endl;

    return cell;
}
