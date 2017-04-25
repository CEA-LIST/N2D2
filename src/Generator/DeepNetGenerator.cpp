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

#include "Generator/DeepNetGenerator.hpp"

std::shared_ptr<N2D2::DeepNet>
N2D2::DeepNetGenerator::generate(Network& network, const std::string& fileName)
{
    IniParser iniConfig;

    std::cout << "Loading network configuration file " << fileName << std::endl;
    iniConfig.load(fileName);

    // Global parameters
    iniConfig.currentSection();
    CellGenerator::mDefaultModel = iniConfig.getProperty
                                   <std::string>("DefaultModel", "Transcode");

#ifndef CUDA
    const std::string suffix = "_CUDA";
    const int compareSize = std::max<size_t>(CellGenerator::mDefaultModel.size()
                                     - suffix.size(), 0);

    if (CellGenerator::mDefaultModel.compare(compareSize, suffix.size(), suffix)
        == 0)
    {
        std::cout << Utils::cwarning << "Warning: to use "
            << CellGenerator::mDefaultModel << " models, N2D2 must be compiled "
            "with CUDA enabled.\n";

        CellGenerator::mDefaultModel
            = CellGenerator::mDefaultModel.substr(0, compareSize);

        std::cout << "*** Using " << CellGenerator::mDefaultModel
            << " model instead. ***" << Utils::cdef << std::endl;
    }
#endif

    if (CellGenerator::mDefaultModel == "RRAM") {
        Synapse_RRAM::setProgramMethod(iniConfig.getProperty(
            "ProgramMethod(" + CellGenerator::mDefaultModel + ")",
            Synapse_RRAM::Ideal));
    } else if (CellGenerator::mDefaultModel == "PCM") {
        Synapse_PCM::setProgramMethod(iniConfig.getProperty(
            "ProgramMethod(" + CellGenerator::mDefaultModel + ")",
            Synapse_PCM::Ideal));
    }

    iniConfig.ignoreProperty("ProgramMethod(*)");

    Synapse_Static::setCheckWeightRange(iniConfig.getProperty
                                        <bool>("CheckWeightRange", true));

    std::shared_ptr<DeepNet> deepNet(new DeepNet(network));
    deepNet->setSignalsDiscretization(
        iniConfig.getProperty<unsigned int>("SignalsDiscretization", 0U));
    deepNet->setFreeParametersDiscretization(
        iniConfig.getProperty
        <unsigned int>("FreeParametersDiscretization", 0U));

    if (iniConfig.isSection("database"))
        deepNet->setDatabase(
            DatabaseGenerator::generate(iniConfig, "database"));
    else {
        std::cout << Utils::cwarning << "Warning: no database specified."
                  << Utils::cdef << std::endl;
        deepNet->setDatabase(std::make_shared<Database>());
    }

    // Set up the environment
    if (iniConfig.isSection("cenv"))
        deepNet->setStimuliProvider(CEnvironmentGenerator::generate(
            *deepNet->getDatabase(), iniConfig, "cenv"));
    else if (iniConfig.isSection("env"))
        deepNet->setStimuliProvider(EnvironmentGenerator::generate(
            network, *deepNet->getDatabase(), iniConfig, "env"));
    else
        deepNet->setStimuliProvider(StimuliProviderGenerator::generate(
            *deepNet->getDatabase(), iniConfig, "sp"));

    // Construct network tree
    // std::cout << "Construct network tree..." << std::endl;
    std::map<std::string, std::vector<std::string> > parentLayers;

    const std::vector<std::string> sections = iniConfig.getSections();

    for (std::vector<std::string>::const_iterator itSection = sections.begin(),
                                                  itSectionEnd = sections.end();
         itSection != itSectionEnd;
         ++itSection) {
        iniConfig.currentSection(*itSection, false);

        if (iniConfig.isProperty("Input")) {
            std::vector<std::string> inputs = Utils::split(
                iniConfig.getProperty<std::string>("Input"), ",");

            std::map<std::string, std::vector<std::string> >::iterator
                itParent;
            std::tie(itParent, std::ignore) = parentLayers.insert(
                std::make_pair((*itSection), std::vector<std::string>()));

            for (std::vector<std::string>::iterator it = inputs.begin(),
                                                    itEnd = inputs.end();
                 it != itEnd;
                 ++it)
            {
                if ((*it) == "sp" || (*it) == "cenv")
                    (*it) = "env";

                (*itParent).second.push_back((*it));
                // std::cout << "  " << (*it) << " => " << (*itSection) <<
                // std::endl;
            }
        }
    }

    std::vector<std::vector<std::string> > layers(
        1, std::vector<std::string>(1, "env"));

    std::map<std::string, unsigned int> layersOrder;
    layersOrder.insert(std::make_pair("env", 0));
    unsigned int nbOrderedLayers = 0;
    unsigned int nbOrderedLayersNext = 1;

    while (nbOrderedLayers < nbOrderedLayersNext) {
        nbOrderedLayers = nbOrderedLayersNext;

        for (std::map<std::string, std::vector<std::string> >::const_iterator it
             = parentLayers.begin(),
             itEnd = parentLayers.end();
             it != itEnd;
             ++it)
        {
            unsigned int order = 0;
            bool knownOrder = true;

            for (std::vector<std::string>::const_iterator itParent
                 = (*it).second.begin();
                 itParent != (*it).second.end();
                 ++itParent)
            {
                const std::map
                    <std::string, unsigned int>::const_iterator itLayer
                    = layersOrder.find((*itParent));

                if (itLayer != layersOrder.end())
                    order = std::max(order, (*itLayer).second);
                else {
                    knownOrder = false;
                    break;
                }
            }

            if (knownOrder) {
                layersOrder.insert(std::make_pair((*it).first, order + 1));

                if (order + 1 >= layers.size())
                    layers.resize(order + 2);

                if (std::find(layers[order + 1].begin(),
                              layers[order + 1].end(),
                              (*it).first) == layers[order + 1].end()) {
                    layers[order + 1].push_back((*it).first);
                    // std::cout << "  " << (*it).first << " = " << order + 1 <<
                    // std::endl;

                    ++nbOrderedLayersNext;
                }
            }
        }
    }

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it)
        {
            std::vector<std::shared_ptr<Cell> > parentCells;

            for (std::vector<std::string>::const_iterator itParent
                 = parentLayers[(*it)].begin();
                 itParent != parentLayers[(*it)].end();
                 ++itParent)
            {
                if ((*itParent) == "env")
                    parentCells.push_back(std::shared_ptr<Cell>());
                else
                    parentCells.push_back(deepNet->getCell((*itParent)));
            }

            // Set up the layer
            std::shared_ptr<Cell> cell
                = CellGenerator::generate(network,
                                          *deepNet->getStimuliProvider(),
                                          parentCells,
                                          iniConfig,
                                          *it);
            deepNet->addCell(cell, parentCells);

            const std::vector<std::string> targets
                = iniConfig.getSections((*it) + ".Target*");

            for (std::vector<std::string>::const_iterator itTarget
                 = targets.begin(),
                 itTargetEnd = targets.end();
                 itTarget != itTargetEnd;
                 ++itTarget) {
                std::shared_ptr<Target> target = TargetGenerator::generate(
                    cell, deepNet, iniConfig, (*itTarget));
                deepNet->addTarget(target);
            }

            // Monitor for the cell
            std::shared_ptr<Cell_Spike> cellSpike = std::dynamic_pointer_cast
                <Cell_Spike>(cell);

            if (cellSpike) {
                std::shared_ptr<Monitor> monitor(new Monitor(network));
                monitor->add(cellSpike->getOutputs());

                deepNet->addMonitor((*it), monitor);
            }
        }
    }

    if (deepNet->getTargets().empty())
        throw std::runtime_error(
            "Missing target cell (no [*.Target] section found)");

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator itCells
         = deepNet->getCells().begin(),
         itCellsEnd = deepNet->getCells().end();
         itCells != itCellsEnd;
         ++itCells) {
        CellGenerator::postGenerate(
            (*itCells).second, deepNet, iniConfig, (*itCells).first);
    }

    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = deepNet->getTargets().begin(),
         itTargetsEnd = deepNet->getTargets().end();
         itTargets != itTargetsEnd;
         ++itTargets) {
        TargetGenerator::postGenerate(
            (*itTargets), deepNet, iniConfig, (*itTargets)->getName());
    }

    // Monitor for the environment
    std::shared_ptr<Environment> env = std::dynamic_pointer_cast
        <Environment>(deepNet->getStimuliProvider());

    if (env) {
        std::shared_ptr<Monitor> monitor(new Monitor(network));
        monitor->add(env->getNodes());

        deepNet->addMonitor("env", monitor);
    }

    // Check that the properties of the latest section are valid
    iniConfig.currentSection();

    Cell::Stats stats;
    deepNet->getStats(stats);

    std::cout << "Total number of neurons: " << stats.nbNeurons << std::endl;
    std::cout << "Total number of nodes: " << stats.nbNodes << std::endl;
    std::cout << "Total number of synapses: " << stats.nbSynapses << std::endl;
    std::cout << "Total number of virtual synapses: " << stats.nbVirtualSynapses
              << std::endl;
    std::cout << "Total number of connections: " << stats.nbConnections
              << std::endl;

    return deepNet;
}
