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

#include "DeepNet.hpp"
#include "Generator/DropoutCellGenerator.hpp"
#include "third_party/half.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::DropoutCellGenerator::mRegistrar(DropoutCell::Type,
                                       N2D2::DropoutCellGenerator::generate);

std::shared_ptr<N2D2::DropoutCell>
N2D2::DropoutCellGenerator::generate(Network& /*network*/, const DeepNet& deepNet,
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

    std::cout << "Layer: " << section << " [Dropout(" << model << ")]"
              << std::endl;

    // Cell construction
    std::shared_ptr<DropoutCell> cell
        = (dataType == Float32)
            ? Registrar<DropoutCell>::create<float>(model)(deepNet, section, nbOutputs)
          : (dataType == Float16)
            ? Registrar<DropoutCell>::create<half_float::half>(model)(deepNet, section,
                                                                      nbOutputs)
            : Registrar<DropoutCell>::create<double>(model)(deepNet, section,
                                                            nbOutputs);

    if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in network configuration file: " + iniConfig.getFileName());
    }

    // Set configuration parameters defined in the INI file
    generateParams(cell, iniConfig, section, model, dataType);

    const unsigned int x0 = iniConfig.getProperty
                            <unsigned int>("InputOffsetX", 0);
    const unsigned int y0 = iniConfig.getProperty
                            <unsigned int>("InputOffsetY", 0);
    const unsigned int width = iniConfig.getProperty
                               <unsigned int>("InputWidth", 0);
    const unsigned int height = iniConfig.getProperty
                                <unsigned int>("InputHeight", 0);

    // Connect the cell to the parents
    for (std::vector<std::shared_ptr<Cell> >::const_iterator it
         = parents.begin(),
         itEnd = parents.end();
         it != itEnd;
         ++it) {
        if (!(*it))
            cell->addInput(sp, x0, y0, width, height);
        else
            cell->addInput((*it).get(), x0, y0, width, height);
    }

    std::cout << "  # Inputs dims: " << cell->getInputsDims() << std::endl;
    std::cout << "  # Outputs dims: " << cell->getOutputsDims() << std::endl;

    return cell;
}

void N2D2::DropoutCellGenerator::generateParams(const std::shared_ptr<DropoutCell>& cell,
                                  IniParser& iniConfig,
                                  const std::string& section,
                                  const std::string& model,
                                  const DataType& /*dataType*/)
{
    cell->setParameters(getConfig(model, iniConfig));

    // Load configuration file (if exists)
    cell->loadParameters(section + ".cfg", true);
}
