/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include <memory>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DeepNet.hpp"
#include "Xnet/Network.hpp"
#include "StimuliProvider.hpp"
#include "Cell/Cell.hpp"
#include "Cell/NormalizeCell.hpp"
#include "Generator/CellGenerator.hpp"
#include "Generator/NormalizeCellGenerator.hpp"
#include "utils/IniParser.hpp"


static const N2D2::Registrar<N2D2::CellGenerator> registrar(N2D2::NormalizeCell::Type, 
                                                            N2D2::NormalizeCellGenerator::generate);

std::shared_ptr<N2D2::NormalizeCell> N2D2::NormalizeCellGenerator::generate(Network& /*network*/, 
                                        const DeepNet& deepNet, StimuliProvider& sp,
                                        const std::vector<std::shared_ptr<Cell>>& parents,
                                        IniParser& iniConfig, const std::string& section)
{
    if (!iniConfig.currentSection(section, false)) {
        throw std::runtime_error("Missing [" + section + "] section.");
    }

    const std::string model = iniConfig.getProperty<std::string>(
        "Model", CellGenerator::mDefaultModel);
    const DataType dataType = iniConfig.getProperty<DataType>(
        "DataType", CellGenerator::mDefaultDataType);

    const unsigned int nbOutputs = iniConfig.getProperty<unsigned int>("NbOutputs");
    const NormalizeCell::Norm norm = iniConfig.getProperty<NormalizeCell::Norm>("Norm");

    std::cout << "Layer: " << section << " [Normalize(" << model << ")]" << std::endl;   

    std::shared_ptr<NormalizeCell> cell
        = (dataType == Float32)
            ?  Registrar<NormalizeCell>::create<float>(model)(deepNet, section,
                                                              nbOutputs, norm)
        : (dataType == Float16) ?
                Registrar<NormalizeCell>::create<half_float::half>(model)(deepNet, section,
                                                              nbOutputs, norm)
        :
                Registrar<NormalizeCell>::create<double>(model)(deepNet, section,
                                                              nbOutputs, norm);

    if (!cell) {
        throw std::runtime_error("Cell model \"" + model + "\" is not valid in section [" + section + "] "
                                 "in network configuration file: " + iniConfig.getFileName());
    }

    // Set configuration parameters defined in the INI file
    cell->setParameters(getConfig(model, iniConfig));

    // Load configuration file (if exists)
    cell->loadParameters(section + ".cfg", true);

    // Connect the cell to the parents
    for (const auto& parent: parents) {
        if (!parent) {
            cell->addInput(sp, 0, 0, sp.getSizeX(), sp.getSizeY());
        }
        else {
            cell->addInput(parent.get());
        }
    }
    
    std::cout << "  # Inputs dims: " << cell->getInputsDims() << std::endl;
    std::cout << "  # Outputs dims: " << cell->getOutputsDims() << std::endl;

    return cell;
}