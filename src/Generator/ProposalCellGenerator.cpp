/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Generator/ProposalCellGenerator.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::ProposalCellGenerator::mRegistrar(ProposalCell::Type,
                                       N2D2::ProposalCellGenerator::generate);

std::shared_ptr<N2D2::ProposalCell>
N2D2::ProposalCellGenerator::generate(Network& /*network*/,
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


    const unsigned int nbProposals = iniConfig.getProperty
                                   <unsigned int>("NbProposals");
    const unsigned int scoreIndex = iniConfig.getProperty
                                   <unsigned int>("ScoreIndex", 0);
    const unsigned int IoUIndex = iniConfig.getProperty
                                   <unsigned int>("IoUIndex", 5);            
    const bool withNMS = iniConfig.getProperty
                                   <bool>("ApplyNMS", false);

    const bool withCls = iniConfig.getProperty
                                   <bool>("WithCls", false);

    const std::vector<double> meansFactor = iniConfig.getProperty
                <std::vector<double> >("MeansFactor", std::vector<double>(4, 0.0));

    const std::vector<double> stdFactor = iniConfig.getProperty
                <std::vector<double> >("StdFactor", std::vector<double>(4, 1.0));

    if(meansFactor.size() != 4 )
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [MeansFactor]"
            " in network configuration file: " + iniConfig.getFileName() + " MeanFactor must have a size of 4");

    if(stdFactor.size() != 4 )
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [StdFactor]"
            " in network configuration file: " + iniConfig.getFileName() + " StdFactor must have a size of 4");

    const std::vector<unsigned int> partsPerCls = iniConfig.getProperty
                <std::vector<unsigned int> >("NumParts", std::vector<unsigned int>(0, 0));

    const std::vector<unsigned int> templatesPerCls = iniConfig.getProperty
                <std::vector<unsigned int> >("NumTemplates", std::vector<unsigned int>(0, 0));

    std::cout << "Layer: " << section << " [Proposal(" << model << ")]"
              << std::endl;
    unsigned int nbOutputParts = 0;
    unsigned int nbOutputTemplates = 0;

    if(partsPerCls.size() > 0)
        nbOutputParts = (*std::max_element(partsPerCls.begin(), partsPerCls.end()))*2;

    if(templatesPerCls.size() > 0)
        nbOutputTemplates = (*std::max_element(templatesPerCls.begin(), templatesPerCls.end()))*3;

    const unsigned int nbOutputs = withCls ? 5 + nbOutputParts + nbOutputTemplates : 4;
    // Cell construction
    std::shared_ptr<ProposalCell> cell = Registrar
        <ProposalCell>::create(model)(section,
                                sp,
                                nbOutputs,
                                nbProposals,
                                scoreIndex,
                                IoUIndex,
                                withNMS,
                                meansFactor,
                                stdFactor,
                                partsPerCls,
                                templatesPerCls);

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
