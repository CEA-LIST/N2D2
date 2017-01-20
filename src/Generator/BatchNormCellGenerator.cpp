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
