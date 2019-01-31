/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): Thibault ALLENET (thibault.allenet@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
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
#include "Generator/LSTMCellGenerator.hpp"
#include "Solver/Solver.hpp"
#include "third_party/half.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::LSTMCellGenerator::mRegistrar(LSTMCell::Type,
                                    N2D2::LSTMCellGenerator::generate);

std::shared_ptr<N2D2::LSTMCell>
N2D2::LSTMCellGenerator::generate(Network& network,
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

	std::cout << "Layer: " << section << " [LSTM(" << model << ")]"
				<< std::endl;

	const unsigned int seqLength = iniConfig.getProperty
									<unsigned int>("SeqLength");
	const unsigned int batchSize = iniConfig.getProperty
									<unsigned int>("BatchSize");
	const unsigned int inputDim = iniConfig.getProperty
									<unsigned int>("InputDim");
	const unsigned int numberLayers = iniConfig.getProperty
									<unsigned int>("NumberLayers");
	const unsigned int hiddenSize = iniConfig.getProperty
									<unsigned int>("HiddenSize");

	unsigned int algo, bidirectional,inputMode;

	float dropout;

	bool singleBackpropFeeding;

	if (iniConfig.isProperty("Algo")){
		algo = iniConfig.getProperty<unsigned int>("Algo");
	}else {
		algo = iniConfig.getProperty<unsigned int>("Algo", 0); //Algo STANDARD default
	}

	if (iniConfig.isProperty("Bidirectional")){
		bidirectional = iniConfig.getProperty<unsigned int>("Bidirectional");
	}else {
		bidirectional = iniConfig.getProperty<unsigned int>("Bidirectional", 0); //UNIDIRECTIONNAL default
	}

	if (iniConfig.isProperty("InputMode")){
		inputMode = iniConfig.getProperty<unsigned int>("InputMode");
	}else {
		inputMode = iniConfig.getProperty<unsigned int>("InputMode", 1); //LINEAR default
	}

	if (iniConfig.isProperty("Dropout")){
		dropout = iniConfig.getProperty<float>("Dropout");
	}else {
		dropout = iniConfig.getProperty<float>("Dropout", 0.); //No dropout default
	}

	if (iniConfig.isProperty("SingleBackpropFeeding")){
		singleBackpropFeeding = iniConfig.getProperty<bool>("SingleBackpropFeeding");
	}else {
		singleBackpropFeeding = iniConfig.getProperty<bool>("SingleBackpropFeeding", true); //Default backpropagation gradient feeding set to all timeStep
	}
	std::cout << "-------------------> Arguments collected " << std::endl;

	std::shared_ptr<LSTMCell> cell
        = (dataType == Float32)
            ? Registrar<LSTMCell>::create<float>(model)(network,
														section,
														seqLength,
														batchSize,
														inputDim,
														numberLayers,
														hiddenSize,
														algo,
														batchSize,
														bidirectional,
														inputMode,
														dropout,
														singleBackpropFeeding)
          : (dataType == Float16)
            ? Registrar<LSTMCell>::create<half_float::half>(model)(network,
																	section,
																	seqLength,
																	batchSize,
																	inputDim,
																	numberLayers,
																	hiddenSize,
																	algo,
																	batchSize,
																	bidirectional,
																	inputMode,
																	dropout,
																	singleBackpropFeeding)
            : Registrar<LSTMCell>::create<double>(model)(network,
															section,
															seqLength,
															batchSize,
															inputDim,
															numberLayers,
															hiddenSize,
															algo,
															batchSize,
															bidirectional,
															inputMode,
															dropout,
															singleBackpropFeeding);

	std::cout << "------------------->LSTM Cell created " << std::endl;

	if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in configuration file: " + iniConfig.getFileName());
	}

	// Set configuration parameters defined in the INI file

	 std::shared_ptr<Solver> solvers
        = (dataType == Float16)
            ? SolverGenerator::generate(iniConfig, section, model, Float32, "Solvers")
            : SolverGenerator::generate(iniConfig, section, model, dataType, "Solvers");

	if (solvers) {
		cell->setWeightsSolver(solvers);
		}

	std::map<std::string, std::string> params = getConfig(model, iniConfig);

	if (cell->getWeightsSolver()) {
		std::cout << "------------------->setting parameters for solver!!!!!!!! " << std::endl;
		cell->getWeightsSolver()->setPrefixedParameters(params, "Solvers.");
	}

	std::cout << "------------------->LSTM Solvers created " << std::endl;
	cell->setParameters(params);

	cell->loadParameters(section + ".cfg", true);



	// Set Gate fillers
	if (iniConfig.isProperty("AllGatesWeightsFiller")){
		cell->setWeightsPreviousLayerAllGateFiller_1stLayer(FillerGenerator::generate(iniConfig, section, "AllGatesWeightsFiller", dataType));
		cell->setWeightsPreviousLayerAllGateFiller(FillerGenerator::generate(iniConfig, section, "AllGatesWeightsFiller", dataType));
		cell->setWeightsRecurrentAllGateFiller(FillerGenerator::generate(iniConfig, section, "AllGatesWeightsFiller", dataType));
	}

	if (iniConfig.isProperty("AllGatesBiasFiller")){
		cell->setBiasAllGateFiller(FillerGenerator::generate(iniConfig, section, "AllGatesBiasFiller", dataType));
	}

	if (iniConfig.isProperty("WeightsInputGateFiller")){
		cell->setWeightsPreviousLayerInputGateFiller_1stLayer(FillerGenerator::generate(iniConfig, section, "WeightsInputGateFiller", dataType));
		cell->setWeightsPreviousLayerInputGateFiller(FillerGenerator::generate(iniConfig, section, "WeightsInputGateFiller", dataType));
		cell->setWeightsRecurrentInputGateFiller(FillerGenerator::generate(iniConfig, section, "WeightsInputGateFiller", dataType));
	}

	if (iniConfig.isProperty("WeightsForgetGateFiller")){
		cell->setWeightsPreviousLayerForgetGateFiller_1stLayer(FillerGenerator::generate(iniConfig, section, "WeightsForgetGateFiller", dataType));
		cell->setWeightsPreviousLayerForgetGateFiller(FillerGenerator::generate(iniConfig, section, "WeightsForgetGateFiller", dataType));
		cell->setWeightsRecurrentForgetGateFiller(FillerGenerator::generate(iniConfig, section, "WeightsForgetGateFiller", dataType));
	}

	if (iniConfig.isProperty("WeightsCellGateFiller")){
		cell->setWeightsPreviousLayerCellGateFiller_1stLayer(FillerGenerator::generate(iniConfig, section, "WeightsCellGateFiller", dataType));
		cell->setWeightsPreviousLayerCellGateFiller(FillerGenerator::generate(iniConfig, section, "WeightsCellGateFiller", dataType));
		cell->setWeightsRecurrentCellGateFiller(FillerGenerator::generate(iniConfig, section, "WeightsCellGateFiller", dataType));
	}

	if (iniConfig.isProperty("WeightsOutputGateFiller")){
		cell->setWeightsPreviousLayerOutputGateFiller_1stLayer(FillerGenerator::generate(iniConfig, section, "WeightsOutputGateFiller", dataType));
		cell->setWeightsPreviousLayerOutputGateFiller(FillerGenerator::generate(iniConfig, section, "WeightsOutputGateFiller", dataType));
		cell->setWeightsRecurrentOutputGateFiller(FillerGenerator::generate(iniConfig, section, "WeightsOutputGateFiller", dataType));
	}


	if (iniConfig.isProperty("BiasInputGateFiller")){
		cell->setBiasPreviousLayerInputGateFiller(FillerGenerator::generate(iniConfig, section, "BiasInputGateFiller", dataType));
		cell->setBiasRecurrentInputGateFiller(FillerGenerator::generate(iniConfig, section, "BiasInputGateFiller", dataType));
	}

	if (iniConfig.isProperty("BiasReccurentForgetGateFiller")){
		cell->setBiasRecurrentForgetGateFiller(FillerGenerator::generate(iniConfig, section, "BiasReccurentForgetGateFiller", dataType));
	}
	if (iniConfig.isProperty("BiasPreviousLayerForgetGateFiller")){
		cell->setBiasPreviousLayerForgetGateFiller(FillerGenerator::generate(iniConfig, section, "BiasPreviousLayerForgetGateFiller", dataType));
	}

	if (iniConfig.isProperty("BiasCellGateFiller")){
		cell->setBiasPreviousLayerCellGateFiller(FillerGenerator::generate(iniConfig, section, "BiasCellGateFiller", dataType));
		cell->setBiasRecurrentCellGateFiller(FillerGenerator::generate(iniConfig, section, "BiasCellGateFiller", dataType));
	}

	if (iniConfig.isProperty("BiasOutputGateFiller")){
		cell->setBiasPreviousLayerOutputGateFiller(FillerGenerator::generate(iniConfig, section, "BiasOutputGateFiller", dataType));
		cell->setBiasRecurrentOutputGateFiller(FillerGenerator::generate(iniConfig, section, "BiasOutputGateFiller", dataType));
	}

	// Set hprev fillers

	if (iniConfig.isProperty("HxFiller"))
		cell->setHxFiller(FillerGenerator::generate(iniConfig, section, "HxFiller", dataType));

	// Set cprev fillers

	if (iniConfig.isProperty("CxFiller"))
		cell->setCxFiller(FillerGenerator::generate(iniConfig, section, "CxFiller", dataType));

	std::cout << "------------------->LSTM Fillers created " << std::endl;
	// Connect the cell to the parents

   	for (std::vector<std::shared_ptr<Cell> >::const_iterator it
         = parents.begin(),
         itEnd = parents.end();
         it != itEnd;
         ++it) {
        if (!(*it)){
            cell->addInput(sp, 0, 0, sp.getSizeX(), sp.getSizeY());
		}
        else {
            cell->addInput((*it).get());
		}
    }

	std::cout << "Layer: " << section << " [LSTM(" << model << ")] generation done"
				<< std::endl;

	return cell;
}



