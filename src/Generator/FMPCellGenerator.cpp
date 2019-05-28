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
#include "Generator/FMPCellGenerator.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::FMPCellGenerator::mRegistrar(FMPCell::Type,
                                   N2D2::FMPCellGenerator::generate);

std::shared_ptr<N2D2::FMPCell>
N2D2::FMPCellGenerator::generate(Network& network, const DeepNet& deepNet,
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

    std::cout << "Layer: " << section << " [FMP(" << model << ")]" << std::endl;

    const unsigned int nbOutputs = iniConfig.getProperty
                                    <unsigned int>("NbOutputs");
    const double scalingRatio = iniConfig.getProperty<double>("ScalingRatio");

    std::shared_ptr<Activation> activation
        = ActivationGenerator::generate(
            iniConfig, section, model, dataType, "ActivationFunction");

    // Cell construction
    std::shared_ptr<FMPCell> cell = Registrar<FMPCell>::create(model)(
        network, deepNet, section, scalingRatio, nbOutputs, activation);

    if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in network configuration file: " + iniConfig.getFileName());
    }

    // Set configuration parameters defined in the INI file
    cell->setParameters(getConfig(model, iniConfig));

    // Load configuration file (if exists)
    cell->loadParameters(section + ".cfg", true);

    // Connect the cell to the parents
    MappingGenerator::Mapping defaultMapping
        = MappingGenerator::getMapping(iniConfig, section, "Mapping");

    std::vector<bool> outputConnection(nbOutputs, false);

    for (std::vector<std::shared_ptr<Cell> >::const_iterator it
         = parents.begin(),
         itEnd = parents.end();
         it != itEnd;
         ++it) {
        const Tensor<bool> map = MappingGenerator::generate(
            sp, (*it), nbOutputs, iniConfig, section, "Mapping",
                                                            defaultMapping);

        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int channel = 0; channel < map.dimY(); ++channel) {
                outputConnection[output] = outputConnection[output]
                                            || map(output, channel);
            }
        }

        if (!(*it))
            cell->addInput(sp, 0, 0, sp.getSizeX(), sp.getSizeY(), map);
        else if ((*it)->getOutputsWidth() > 1 || (*it)->getOutputsHeight() > 1)
            cell->addInput((*it).get(), map);
        else {
            throw std::runtime_error("2D input expected for a FMPCell (\""
                                     + section + "\"), \"" + (*it)->getName()
                                     + "\" is not,"
                                       " in configuration file: "
                                     + iniConfig.getFileName());
        }
    }

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        if (!outputConnection[output]) {
            std::cout << Utils::cwarning << "Warning: output map #" << output
                      << " of \"" << section << "\" has no input connection."
                      << Utils::cdef << std::endl;
        }
    }

    std::cout << "  # Inputs dims: " << cell->getInputsDims() << std::endl;
    std::cout << "  # Outputs dims: " << cell->getOutputsDims() << std::endl;

    Utils::createDirectories("map");
    cell->writeMap("map/" + section + "_map.dat");

    return cell;
}
