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

#include "Generator/BatchNormCellGenerator.hpp"

N2D2::Registrar<N2D2::CellGenerator> N2D2::BatchNormCellGenerator::mRegistrar(
    BatchNormCell::Type, N2D2::BatchNormCellGenerator::generate);
N2D2::RegistrarCustom<N2D2::CellGenerator,
N2D2::CellGenerator::RegistryPostCreate_T>
N2D2::BatchNormCellGenerator::mRegistrarPost(BatchNormCell::Type
    + std::string("+"), N2D2::BatchNormCellGenerator::postGenerate);

std::shared_ptr<N2D2::BatchNormCell>
N2D2::BatchNormCellGenerator::generate(Network& /*network*/,
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
    const unsigned int nbOutputs = iniConfig.getProperty
                                   <unsigned int>("NbOutputs");

    std::cout << "Layer: " << section << " [BatchNorm(" << model << ")]"
              << std::endl;

    std::shared_ptr<Activation<Float_T> > activation
        = ActivationGenerator::generate(
            iniConfig,
            section,
            model,
            "ActivationFunction",
            Registrar<TanhActivation<Float_T> >::create(model)());

    // Cell construction
    std::shared_ptr<BatchNormCell> cell = Registrar
        <BatchNormCell>::create(model)(section, nbOutputs, activation);

    if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in network configuration file: " + iniConfig.getFileName());
    }

    // Set configuration parameters defined in the INI file
    std::shared_ptr<Solver<Float_T> > solvers
        = SolverGenerator::generate(iniConfig, section, model, "Solvers");

    if (solvers) {
        cell->setScaleSolver(solvers);
        cell->setBiasSolver(solvers->clone());
    }

    std::shared_ptr<Solver<Float_T> > scaleSolver
        = SolverGenerator::generate(iniConfig, section, model, "ScaleSolver");

    if (scaleSolver)
        cell->setScaleSolver(scaleSolver);

    std::shared_ptr<Solver<Float_T> > biasSolver
        = SolverGenerator::generate(iniConfig, section, model, "BiasSolver");

    if (biasSolver)
        cell->setBiasSolver(biasSolver);

    std::map<std::string, std::string> params = getConfig(model, iniConfig);

    if (cell->getScaleSolver()) {
        cell->getScaleSolver()->setPrefixedParameters(
            params, "Solvers.", false);
        cell->getScaleSolver()->setPrefixedParameters(params, "ScaleSolver.");
    }

    if (cell->getBiasSolver()) {
        cell->getBiasSolver()->setPrefixedParameters(params, "Solvers.");
        cell->getBiasSolver()->setPrefixedParameters(params, "BiasSolver.");
    }

    // Will be processed in postGenerate
    iniConfig.ignoreProperty("ScalesSharing");
    iniConfig.ignoreProperty("BiasesSharing");
    iniConfig.ignoreProperty("MeansSharing");
    iniConfig.ignoreProperty("VariancesSharing");

    cell->setParameters(params);

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
            throw std::runtime_error("2D input expected for a BatchNormCell (\""
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

void N2D2::BatchNormCellGenerator::postGenerate(const std::shared_ptr<Cell>& cell,
                                             const std::shared_ptr
                                             <DeepNet>& deepNet,
                                             IniParser& iniConfig,
                                             const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    std::shared_ptr<BatchNormCell> batchNormCell
        = std::dynamic_pointer_cast<BatchNormCell>(cell);

    if (iniConfig.isProperty("ScalesSharing")) {
        const std::string scalesSharing
            = iniConfig.getProperty<std::string>("ScalesSharing");

        bool found = false;

        for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator
            itCells = deepNet->getCells().begin(),
             itCellsEnd = deepNet->getCells().end();
             itCells != itCellsEnd;
             ++itCells)
        {
            if ((*itCells).first == scalesSharing) {
                std::shared_ptr<BatchNormCell> cellRef
                    = std::dynamic_pointer_cast<BatchNormCell>(
                                                    (*itCells).second);

                if (!cellRef) {
                    throw std::runtime_error("Cell name \"" + scalesSharing
                                             + "\" is not a BatchNormCell for"
                                             " ScalesSharing [" + section
                                             + "] in network configuration "
                                             + "file: "
                                             + iniConfig.getFileName());
                }

                batchNormCell->setScales(cellRef->getScales());

                std::cout << Utils::cnotice << "Sharing scales from cell "
                    << scalesSharing << " for cell " << section << Utils::cdef
                    << std::endl;

                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("Cell name \"" + scalesSharing
                                     + "\" not found for ScalesSharing ["
                                     + section
                                     + "] in network configuration file: "
                                     + iniConfig.getFileName());
        }
    }

    if (iniConfig.isProperty("BiasesSharing")) {
        const std::string biasesSharing
            = iniConfig.getProperty<std::string>("BiasesSharing");

        bool found = false;

        for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator
            itCells = deepNet->getCells().begin(),
             itCellsEnd = deepNet->getCells().end();
             itCells != itCellsEnd;
             ++itCells)
        {
            if ((*itCells).first == biasesSharing) {
                std::shared_ptr<BatchNormCell> cellRef
                    = std::dynamic_pointer_cast<BatchNormCell>(
                                                    (*itCells).second);

                if (!cellRef) {
                    throw std::runtime_error("Cell name \"" + biasesSharing
                                             + "\" is not a BatchNormCell for"
                                             " BiasesSharing [" + section
                                             + "] in network configuration "
                                             + "file: "
                                             + iniConfig.getFileName());
                }

                batchNormCell->setBiases(cellRef->getBiases());

                std::cout << Utils::cnotice << "Sharing biases from cell "
                    << biasesSharing << " for cell " << section << Utils::cdef
                    << std::endl;

                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("Cell name \"" + biasesSharing
                                     + "\" not found for BiasesSharing ["
                                     + section
                                     + "] in network configuration file: "
                                     + iniConfig.getFileName());
        }
    }

    if (iniConfig.isProperty("MeansSharing")) {
        const std::string meansSharing
            = iniConfig.getProperty<std::string>("MeansSharing");

        bool found = false;

        for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator
            itCells = deepNet->getCells().begin(),
             itCellsEnd = deepNet->getCells().end();
             itCells != itCellsEnd;
             ++itCells)
        {
            if ((*itCells).first == meansSharing) {
                std::shared_ptr<BatchNormCell> cellRef
                    = std::dynamic_pointer_cast<BatchNormCell>(
                                                    (*itCells).second);

                if (!cellRef) {
                    throw std::runtime_error("Cell name \"" + meansSharing
                                             + "\" is not a BatchNormCell for"
                                             " MeansSharing [" + section
                                             + "] in network configuration "
                                             + "file: "
                                             + iniConfig.getFileName());
                }

                batchNormCell->setMeans(cellRef->getMeans());

                std::cout << Utils::cnotice << "Sharing means from cell "
                    << meansSharing << " for cell " << section << Utils::cdef
                    << std::endl;

                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("Cell name \"" + meansSharing
                                     + "\" not found for MeansSharing ["
                                     + section
                                     + "] in network configuration file: "
                                     + iniConfig.getFileName());
        }
    }

    if (iniConfig.isProperty("VariancesSharing")) {
        const std::string variancesSharing
            = iniConfig.getProperty<std::string>("VariancesSharing");

        bool found = false;

        for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator
            itCells = deepNet->getCells().begin(),
             itCellsEnd = deepNet->getCells().end();
             itCells != itCellsEnd;
             ++itCells)
        {
            if ((*itCells).first == variancesSharing) {
                std::shared_ptr<BatchNormCell> cellRef
                    = std::dynamic_pointer_cast<BatchNormCell>(
                                                    (*itCells).second);

                if (!cellRef) {
                    throw std::runtime_error("Cell name \"" + variancesSharing
                                             + "\" is not a BatchNormCell for"
                                             " VariancesSharing [" + section
                                             + "] in network configuration "
                                             + "file: "
                                             + iniConfig.getFileName());
                }

                batchNormCell->setVariances(cellRef->getVariances());

                std::cout << Utils::cnotice << "Sharing variances from cell "
                    << variancesSharing << " for cell " << section
                    << Utils::cdef << std::endl;

                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("Cell name \"" + variancesSharing
                                     + "\" not found for VariancesSharing ["
                                     + section
                                     + "] in network configuration file: "
                                     + iniConfig.getFileName());
        }
    }
}
