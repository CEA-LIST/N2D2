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

#include "Generator/DistanceCellGenerator.hpp"
#include "DeepNet.hpp"
#include "StimuliProvider.hpp"

N2D2::Registrar<N2D2::CellGenerator> N2D2::DistanceCellGenerator::mRegistrar(
    DistanceCell::Type, N2D2::DistanceCellGenerator::generate);
/*
static const N2D2::Registrar<N2D2::CellGenerator> registrar(N2D2::DistanceCell::Type, 
                                                            N2D2::DistanceCellGenerator::generate);
*/
std::shared_ptr<N2D2::DistanceCell> N2D2::DistanceCellGenerator::generate(Network& /*network*/, 
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
    const Float_T margin = iniConfig.getProperty<Float_T>("Margin");
    const Float_T centercoef = iniConfig.getProperty<Float_T>("CenterCoef");

    std::cout << "Layer: " << section << " [Distance(" << model << ")]" << std::endl;   

    std::shared_ptr<DistanceCell> cell =  Registrar<DistanceCell>::create<float>(model)(deepNet, section,
                                                              nbOutputs, margin, centercoef);

    if (!cell) {
        throw std::runtime_error("Cell model \"" + model + "\" is not valid in section [" + section + "] "
                                 "in network configuration file: " + iniConfig.getFileName());
    }

    // Set configuration parameters defined in the INI file
    generateParams(cell, iniConfig, section, model, dataType);

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

void N2D2::DistanceCellGenerator::generateParams(const std::shared_ptr<DistanceCell>& cell,
                                  IniParser& iniConfig,
                                  const std::string& section,
                                  const std::string& model,
                                  const DataType& dataType)
{
    std::shared_ptr<Solver> solvers
        = (dataType == Float16)
            ? SolverGenerator::generate(iniConfig, section, model, Float32, "Solvers")
            : SolverGenerator::generate(iniConfig, section, model, dataType, "Solvers");


    if (solvers) {
        cell->setWeightsSolver(solvers->clone());
    }

    std::shared_ptr<Solver> centroidSolver
        = (dataType == Float16)
            ? SolverGenerator::generate(iniConfig, section, model, Float32, "CentroidSolver")
            : SolverGenerator::generate(iniConfig, section, model, dataType, "CentroidSolver");

    if (centroidSolver) {
        cell->setWeightsSolver(centroidSolver);
    }

    std::map<std::string, std::string> params = getConfig(model, iniConfig);
    if (cell->getWeightsSolver()) {
        cell->getWeightsSolver()->setPrefixedParameters(
            params, "Solvers.", false);
        cell->getWeightsSolver()->setPrefixedParameters(params, "CentroidSolver.");
    }

    // Set configuration parameters defined in the INI file
    cell->setParameters(params);

    // Load configuration file (if exists)
    cell->loadParameters(section + ".cfg", true);

    // Set fillers
    if (iniConfig.isProperty("WeightsFiller"))
        cell->setWeightsFiller(FillerGenerator::generate(iniConfig, section, "WeightsFiller", dataType));
}