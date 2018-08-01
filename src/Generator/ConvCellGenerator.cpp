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

#include "Generator/ConvCellGenerator.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::ConvCellGenerator::mRegistrar(ConvCell::Type,
                                    N2D2::ConvCellGenerator::generate);
N2D2::Registrar<N2D2::CellGenerator,
N2D2::CellGenerator::RegistryPostCreate_T>
N2D2::ConvCellGenerator::mRegistrarPost(ConvCell::Type + std::string("+"),
                                      N2D2::ConvCellGenerator::postGenerate);

std::shared_ptr<N2D2::ConvCell>
N2D2::ConvCellGenerator::generate(Network& network,
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

    std::cout << "Layer: " << section << " [Conv(" << model << ")]"
              << std::endl;

    std::vector<unsigned int> kernelDims;

    if (iniConfig.isProperty("KernelDims")) {
        kernelDims = iniConfig.getProperty
                                     <std::vector<unsigned int> >("KernelDims");
    }
    else if (iniConfig.isProperty("Kernel"))
        kernelDims.resize(2, iniConfig.getProperty<unsigned int>("Kernel"));
    else {
        kernelDims.push_back(iniConfig.getProperty
                                     <unsigned int>("KernelWidth"));
        kernelDims.push_back(iniConfig.getProperty
                                     <unsigned int>("KernelHeight"));

        if (iniConfig.isProperty("KernelDepth")) {
            kernelDims.push_back(iniConfig.getProperty
                                         <unsigned int>("KernelDepth"));
        }
    }

    const unsigned int nbChannels = iniConfig.getProperty
                                    <unsigned int>("NbChannels");

    std::vector<unsigned int> subSampleDims;

    if (iniConfig.isProperty("SubSampleDims")) {
        subSampleDims = iniConfig.getProperty
                                <std::vector<unsigned int> >("SubSampleDims");
    }
    else if (iniConfig.isProperty("SubSample")) {
        subSampleDims.resize(kernelDims.size(),
                             iniConfig.getProperty<unsigned int>("SubSample"));
    }
    else {
        subSampleDims.push_back(iniConfig.getProperty
                                     <unsigned int>("SubSampleX", 1));
        subSampleDims.push_back(iniConfig.getProperty
                                     <unsigned int>("SubSampleY", 1));
        subSampleDims.push_back(iniConfig.getProperty
                                     <unsigned int>("SubSampleZ", 1));
        subSampleDims.resize(kernelDims.size(), 1);
    }

    std::vector<unsigned int> strideDims;

    if (iniConfig.isProperty("StrideDims")) {
        strideDims = iniConfig.getProperty
                                <std::vector<unsigned int> >("StrideDims");
    }
    else if (iniConfig.isProperty("Stride")) {
        strideDims.resize(kernelDims.size(),
                             iniConfig.getProperty<unsigned int>("Stride"));
    }
    else {
        strideDims.push_back(iniConfig.getProperty
                                     <unsigned int>("StrideX", 1));
        strideDims.push_back(iniConfig.getProperty
                                     <unsigned int>("StrideY", 1));
        strideDims.push_back(iniConfig.getProperty
                                     <unsigned int>("StrideZ", 1));
        strideDims.resize(kernelDims.size(), 1);
    }

    std::vector<int> paddingDims;

    if (iniConfig.isProperty("PaddingDims")) {
        paddingDims = iniConfig.getProperty<std::vector<int> >("PaddingDims");
    }
    else if (iniConfig.isProperty("Padding")) {
        paddingDims.resize(kernelDims.size(),
                           iniConfig.getProperty<int>("Padding"));
    }
    else {
        paddingDims.push_back(iniConfig.getProperty<int>("PaddingX", 0));
        paddingDims.push_back(iniConfig.getProperty<int>("PaddingY", 0));
        paddingDims.push_back(iniConfig.getProperty<int>("PaddingZ", 0));
        paddingDims.resize(kernelDims.size(), 0);
    }

    std::shared_ptr<Activation> activation
        = ActivationGenerator::generate(
            iniConfig,
            section,
            model,
            dataType,
            "ActivationFunction",
            (dataType == Float32)
                ? Registrar<TanhActivation>::create<float>(model)()
                : Registrar<TanhActivation>::create<double>(model)());

    // Cell construction
    std::shared_ptr<ConvCell> cell
        = (dataType == Float32)
            ? Registrar<ConvCell>::create<float>(model)(network,
                                                        section,
                                                        kernelDims,
                                                        nbChannels,
                                                        subSampleDims,
                                                        strideDims,
                                                        paddingDims,
                                                        activation)
            : Registrar<ConvCell>::create<double>(model)(network,
                                                         section,
                                                         kernelDims,
                                                         nbChannels,
                                                         subSampleDims,
                                                         strideDims,
                                                         paddingDims,
                                                         activation);

    if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in configuration file: " + iniConfig.getFileName());
    }

    // Set configuration parameters defined in the INI file
    std::shared_ptr<Solver> solvers = SolverGenerator::generate(iniConfig,
                                                                section,
                                                                model,
                                                                dataType,
                                                                "Solvers");

    if (solvers) {
        cell->setBiasSolver(solvers);
        cell->setWeightsSolver(solvers->clone());
    }

    std::shared_ptr<Solver> biasSolver = SolverGenerator::generate(iniConfig,
                                                                   section,
                                                                   model,
                                                                   dataType,
                                                                "BiasSolver");

    if (biasSolver)
        cell->setBiasSolver(biasSolver);

    std::shared_ptr<Solver> weightsSolver = SolverGenerator::generate(iniConfig,
                                                                      section,
                                                                      model,
                                                                      dataType,
                                                            "WeightsSolver");

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

    // Will be processed in postGenerate
    iniConfig.ignoreProperty("WeightsSharing");
    iniConfig.ignoreProperty("BiasesSharing");

    cell->setParameters(params);

    // Load configuration file (if exists)
    cell->loadParameters(section + ".cfg", true);

    // Set fillers
    if (iniConfig.isProperty("WeightsFiller"))
        cell->setWeightsFiller(
            FillerGenerator::generate(iniConfig, section, "WeightsFiller", dataType));

    if (iniConfig.isProperty("BiasFiller"))
        cell->setBiasFiller(
            FillerGenerator::generate(iniConfig, section, "BiasFiller", dataType));

    // Connect the cell to the parents
    MappingGenerator::Mapping defaultMapping
        = MappingGenerator::getMapping(iniConfig, section, "Mapping");

    unsigned int nbInputChannels = 0;
    std::vector<bool> outputConnection(nbChannels, false);

    for (std::vector<std::shared_ptr<Cell> >::const_iterator it
         = parents.begin(),
         itEnd = parents.end();
         it != itEnd;
         ++it) {
        const Matrix<bool> map = MappingGenerator::generate(
            sp, (*it), nbChannels, iniConfig, section, defaultMapping);

        nbInputChannels += map.rows();

        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            const std::vector<bool>& row = map.col(channel);
            outputConnection[channel]
                = outputConnection[channel]
                  || (std::find(row.begin(), row.end(), true) != row.end());
        }

        if (!(*it))
            cell->addInput(sp, 0, 0, sp.getSizeX(), sp.getSizeY(), map);
        else if ((*it)->getOutputsWidth() > 1 || (*it)->getOutputsHeight() > 1)
            cell->addInput((*it).get(), map);
        else {
            throw std::runtime_error("2D input expected for a ConvCell (\""
                                     + section + "\"), \"" + (*it)->getName()
                                     + "\" is not,"
                                       " in configuration file: "
                                     + iniConfig.getFileName());
        }
    }

    for (unsigned int channel = 0; channel < nbChannels; ++channel) {
        if (!outputConnection[channel]) {
            std::cout << Utils::cwarning << "Warning: output map #" << channel
                      << " of \"" << section << "\" has no input connection."
                      << Utils::cdef << std::endl;
        }
    }

    // Kernels initialization
    KernelGenerator::setDefault(iniConfig, section, "Kernel");
    const bool defaultNormalize = iniConfig.getProperty
                                  <bool>("Kernel.Normalize", false);

    for (unsigned int channel = 0; channel < nbInputChannels; ++channel) {
        for (unsigned int output = 0; output < nbChannels; ++output) {
            std::stringstream kernelIdx;
            kernelIdx << "Kernel[" << channel << "][" << output << "]";

            Kernel<double> kernel = KernelGenerator::generate
                <double>(iniConfig, section, kernelIdx.str());
            const bool normalize = iniConfig.getProperty<bool>(
                kernelIdx.str() + ".Normalize", defaultNormalize);

            if (!kernel.empty())
                cell->setKernel(output, channel, kernel, normalize);
        }
    }

    std::cout << "  # Shared synapses: " << cell->getNbSharedSynapses()
              << std::endl;
    std::cout << "  # Virtual synapses: " << cell->getNbVirtualSynapses()
              << std::endl;
    std::cout << "  # Inputs dims: " << cell->getInputsDims() << std::endl;
    std::cout << "  # Outputs dims: " << cell->getOutputsDims() << std::endl;

    Utils::createDirectories("map");
    cell->writeMap("map/" + section + "_map.dat");

    return cell;
}

void N2D2::ConvCellGenerator::postGenerate(const std::shared_ptr<Cell>& cell,
                                             const std::shared_ptr
                                             <DeepNet>& deepNet,
                                             IniParser& iniConfig,
                                             const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    std::shared_ptr<ConvCell> convCell
        = std::dynamic_pointer_cast<ConvCell>(cell);

    if (iniConfig.isProperty("WeightsSharing")) {
        const std::vector<std::string> weightsSharing
            = Utils::split(iniConfig.getProperty<std::string>("WeightsSharing"),
                           ",");

        const std::locale offsetLocale(std::locale(),
            new Utils::streamIgnore("[]"));

        for (unsigned int k = 0, size = weightsSharing.size(); k < size; ++k) {
            if (weightsSharing[k].empty())
                continue;

            std::stringstream str(weightsSharing[k]);
            str.imbue(offsetLocale);

            std::string cellName;
            unsigned int offset = k;

            if (!(str >> cellName) || (!str.eof() && !(str >> offset))) {
                throw std::runtime_error("Unreadable value for"
                                         " WeightsSharing [" + section
                                         + "] in network configuration file: "
                                         + iniConfig.getFileName());
            }

            bool found = false;

            for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator
                itCells = deepNet->getCells().begin(),
                 itCellsEnd = deepNet->getCells().end();
                 itCells != itCellsEnd;
                 ++itCells)
            {
                if ((*itCells).first == cellName) {
                    std::shared_ptr<ConvCell> cellRef
                        = std::dynamic_pointer_cast<ConvCell>(
                                                        (*itCells).second);

                    if (!cellRef) {
                        throw std::runtime_error("Cell name \"" + cellName
                                                 + "\" is not a ConvCell for"
                                                 " WeightsSharing [" + section
                                                 + "] in network configuration "
                                                 + "file: "
                                                 + iniConfig.getFileName());
                    }

                    convCell->setWeights(k, cellRef->getWeights(), offset);

                    std::cout << Utils::cnotice << "Sharing weights group #"
                        << offset << " from cell " << cellName << " for cell "
                        << section << " weights group #" << k << Utils::cdef
                        << std::endl;

                    found = true;
                    break;
                }
            }

            if (!found) {
                throw std::runtime_error("Cell name \"" + cellName
                                         + "\" not found for WeightsSharing ["
                                         + section
                                         + "] in network configuration file: "
                                         + iniConfig.getFileName());
            }
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
                std::shared_ptr<ConvCell> cellRef
                    = std::dynamic_pointer_cast<ConvCell>(
                                                    (*itCells).second);

                if (!cellRef) {
                    throw std::runtime_error("Cell name \"" + biasesSharing
                                             + "\" is not a ConvCell for"
                                             " BiasesSharing [" + section
                                             + "] in network configuration "
                                             + "file: "
                                             + iniConfig.getFileName());
                }

                convCell->setBiases(cellRef->getBiases());

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
}
