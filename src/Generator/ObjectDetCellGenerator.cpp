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

#include "Generator/ObjectDetCellGenerator.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::ObjectDetCellGenerator::mRegistrar(ObjectDetCell::Type,
                                        N2D2::ObjectDetCellGenerator::generate);

std::shared_ptr<N2D2::ObjectDetCell>
N2D2::ObjectDetCellGenerator::generate(Network& /*network*/,
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

    const unsigned int nbAnchors = iniConfig.getProperty
                                   <unsigned int>("NbAnchors");
    const unsigned int nbProposals = iniConfig.getProperty
                                   <unsigned int>("NbProposals");
    const unsigned int nbCls = iniConfig.getProperty
                                   <unsigned int>("NbClass");

    const Float_T nmsThreshold = iniConfig.getProperty
                                   <Float_T>("NMS_Threshold", 0.5);
    const Float_T scoreThreshold = iniConfig.getProperty
                                   <Float_T>("Score_Threshold");

    const unsigned int nbOutputs = 5;

    std::cout << "Layer: " << section << " [ObjectDet(" << model << ")]" 
              << std::endl;   
    
    // Cell construction
    std::shared_ptr<ObjectDetCell> cell = Registrar
        <ObjectDetCell>::create(model)(section,
                                sp,
                                nbOutputs,
                                nbAnchors,
                                nbProposals,
                                nbCls,
                                nmsThreshold,
                                scoreThreshold);

    if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in network configuration file: " + iniConfig.getFileName());
    }

    // Set configuration parameters defined in the INI file
    cell->setParameters(getConfig(model, iniConfig));

    // Load configuration file (if exists)
    cell->loadParameters(section + ".cfg", true);

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

    std::cout << "  # Outputs: " << cell->getNbOutputs() << std::endl;

    return cell;
}
