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

#include "Generator/PaddingCellGenerator.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::PaddingCellGenerator::mRegistrar(PaddingCell::Type,
                                   N2D2::PaddingCellGenerator::generate);

std::shared_ptr<N2D2::PaddingCell>
N2D2::PaddingCellGenerator::generate(Network& /*network*/,
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

    const int topPad = iniConfig.getProperty
                                   <int>("TopPadding");
    const int botPad = iniConfig.getProperty
                                    <int>("BottomPadding");
    const int leftPad = iniConfig.getProperty
                                   <int>("LeftPadding");
    const int rightPad = iniConfig.getProperty
                                    <int>("RightPadding");

    const unsigned int nbOutputs = iniConfig.getProperty
                                   <unsigned int>("NbOutputs");

    std::cout << "Layer: " << section << " [Padding(" << model << ")]" << std::endl;

    // Cell construction
    std::shared_ptr<PaddingCell> cell = Registrar
        <PaddingCell>::create(model)(section, 
                                     nbOutputs,
                                     topPad,
                                     botPad,
                                     leftPad,
                                     rightPad);

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
            throw std::runtime_error("2D input expected for a PaddingCell (\""
                                     + section + "\"), \"" + (*it)->getName()
                                     + "\" is not,"
                                       " in configuration file: "
                                     + iniConfig.getFileName());
        }
    }
    
    std::cout << "  # Outputs size: " << cell->getOutputsWidth() << "x"
    << cell->getOutputsHeight() << std::endl;

    std::cout << "  # Outputs: " << cell->getNbOutputs() << std::endl;

    return cell;
}
