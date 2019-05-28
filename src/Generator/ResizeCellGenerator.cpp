/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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
#include "Generator/ResizeCellGenerator.hpp"
#include "StimuliProvider.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::ResizeCellGenerator::mRegistrar(ResizeCell::Type,
                                        N2D2::ResizeCellGenerator::generate);

std::shared_ptr<N2D2::ResizeCell>
N2D2::ResizeCellGenerator::generate(Network& /*network*/, const DeepNet& deepNet,
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

    const unsigned int outputWidth = iniConfig.getProperty
                                   <unsigned int>("OutputWidth");
    const unsigned int outputHeight = iniConfig.getProperty
                                   <unsigned int>("OutputHeight");

    const ResizeCell::ResizeMode resizeMode = iniConfig.getProperty
                                                <ResizeCell::ResizeMode>("Mode");

    const unsigned int nbOutputs = iniConfig.getProperty
                                   <unsigned int>("NbOutputs");

    std::cout << "Layer: " << section << " [Resize(" << model << ")]" 
              << std::endl;   
    
    // Cell construction
    std::shared_ptr<ResizeCell> cell = Registrar
        <ResizeCell>::create(model)(deepNet, section,
                                    outputWidth,
                                    outputHeight,
                                    nbOutputs,
                                    resizeMode);

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
    for (std::vector<std::shared_ptr<Cell> >::const_iterator it
         = parents.begin(),
         itEnd = parents.end();
         it != itEnd;
         ++it) {
        if (!(*it))
            cell->addInput(sp, 0, 0, sp.getSizeX(), sp.getSizeY());
        else if ((*it)->getOutputsWidth() > 1 || (*it)->getOutputsHeight() > 1)
            cell->addInput((*it).get());
        else {
            throw std::runtime_error("2D input expected for a ResizeCell (\""
                                     + section + "\"), \"" + (*it)->getName()
                                     + "\" is not, in configuration file: "
                                     + iniConfig.getFileName());
        }    
    }
    
    std::cout << "  # Outputs size: " << cell->getOutputsWidth() << "x"
              << cell->getOutputsHeight() << std::endl;
    std::cout << "  # Outputs: " << cell->getNbOutputs() << std::endl;

    return cell;
}