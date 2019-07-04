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
#include "Generator/FcCellGenerator.hpp"
#include "third_party/half.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::FcCellGenerator::mRegistrar(FcCell::Type,
                                  N2D2::FcCellGenerator::generate);

std::shared_ptr<N2D2::FcCell>
N2D2::FcCellGenerator::generate(Network& network, const DeepNet& deepNet,
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

    const unsigned int nbOutputs = iniConfig.getProperty
                                   <unsigned int>("NbOutputs");

    std::cout << "Layer: " << section << " [Fc(" << model << ")]" << std::endl;

    std::shared_ptr<Activation> activation
        = ActivationGenerator::generate(
            iniConfig,
            section,
            model,
            dataType,
            "ActivationFunction",
            (dataType == Float32)
                ? Registrar<TanhActivation>::create<float>(model)()
            : (dataType == Float16)
                ? Registrar<TanhActivation>::create<half_float::half>(model)()
                : Registrar<TanhActivation>::create<double>(model)());

    // Cell construction
    std::shared_ptr<FcCell> cell
        = (dataType == Float32)
            ? Registrar<FcCell>::create<float>(model)(network, deepNet, 
                                                      section,
                                                      nbOutputs,
                                                      activation)
          : (dataType == Float16)
            ? Registrar<FcCell>::create<half_float::half>(model)(network, deepNet, 
                                                                 section,
                                                                 nbOutputs,
                                                                 activation)
            : Registrar<FcCell>::create<double>(model)(network, deepNet, 
                                                       section,
                                                       nbOutputs,
                                                       activation);

    if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in network configuration file: " + iniConfig.getFileName());
    }

    // Set configuration parameters defined in the INI file
    std::shared_ptr<Solver> solvers
        = SolverGenerator::generate(iniConfig, section, model, dataType, "Solvers");

    if (solvers) {
        cell->setBiasSolver(solvers);
        cell->setWeightsSolver(solvers->clone());
    }

    std::shared_ptr<Solver> biasSolver
        = SolverGenerator::generate(iniConfig, section, model, dataType, "BiasSolver");

    if (biasSolver)
        cell->setBiasSolver(biasSolver);

    std::shared_ptr<Solver> weightsSolver
        = SolverGenerator::generate(iniConfig, section, model, dataType, "WeightsSolver");

    if (weightsSolver)
        cell->setWeightsSolver(weightsSolver);

    std::map<std::string, std::string> params = getConfig(model, iniConfig);

    if (cell->getBiasSolver()) {
        cell->getBiasSolver()->setPrefixedParameters(params, "Solvers.", false);
        cell->getBiasSolver()->setPrefixedParameters(params, "BiasSolver.");
    }

    if (cell->getWeightsSolver()) {
        cell->getWeightsSolver()->setPrefixedParameters(params, "Solvers.");
        cell->getWeightsSolver()->setPrefixedParameters(params,
                                                        "WeightsSolver.");
    }

    cell->setParameters(params);

    // Load configuration file (if exists)
    cell->loadParameters(section + ".cfg", true);

    // Set fillers
    if (iniConfig.isProperty("WeightsFiller"))
        cell->setWeightsFiller(
            FillerGenerator::generate(iniConfig, section, "WeightsFiller", dataType));

    if (iniConfig.isProperty("TopDownWeightsFiller"))
        cell->setTopDownWeightsFiller(
            FillerGenerator::generate(iniConfig, section, "TopDownWeightsFiller", dataType));

    if (iniConfig.isProperty("RecWeightsFiller"))
        cell->setRecWeightsFiller(
            FillerGenerator::generate(iniConfig, section, "RecWeightsFiller", dataType));

    if (iniConfig.isProperty("BiasFiller"))
        cell->setBiasFiller(
            FillerGenerator::generate(iniConfig, section, "BiasFiller", dataType));

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

    std::cout << "  # Synapses: " << cell->getNbSynapses() << std::endl;
    std::cout << "  # Inputs dims: " << cell->getInputsDims() << std::endl;
    std::cout << "  # Outputs dims: " << cell->getOutputsDims() << std::endl;

    Utils::createDirectories("map");
    cell->writeMap("map/" + section + "_map.dat");

    return cell;
}
