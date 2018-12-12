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

#include "Generator/UnpoolCellGenerator.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::UnpoolCellGenerator::mRegistrar(UnpoolCell::Type,
                                      N2D2::UnpoolCellGenerator::generate);
N2D2::Registrar<N2D2::CellGenerator,
N2D2::CellGenerator::RegistryPostCreate_T>
N2D2::UnpoolCellGenerator::mRegistrarPost(UnpoolCell::Type + std::string("+"),
                                      N2D2::UnpoolCellGenerator::postGenerate);

std::shared_ptr<N2D2::UnpoolCell>
N2D2::UnpoolCellGenerator::generate(Network& network,
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

    std::cout << "Layer: " << section << " [Unpool(" << model << ")]"
              << std::endl;

    std::vector<unsigned int> poolDims;

    if (iniConfig.isProperty("PoolDims")) {
        poolDims = iniConfig.getProperty
                                     <std::vector<unsigned int> >("PoolDims");
    }
    else if (iniConfig.isProperty("PoolSize"))
        poolDims.resize(2, iniConfig.getProperty<unsigned int>("PoolSize"));
    else {
        poolDims.push_back(iniConfig.getProperty
                                     <unsigned int>("PoolWidth"));
        poolDims.push_back(iniConfig.getProperty
                                     <unsigned int>("PoolHeight"));

        if (iniConfig.isProperty("PoolDepth")) {
            poolDims.push_back(iniConfig.getProperty
                                         <unsigned int>("PoolDepth"));
        }
    }

    const unsigned int nbOutputs = iniConfig.getProperty
                                    <unsigned int>("NbOutputs");

    std::vector<unsigned int> strideDims;

    if (iniConfig.isProperty("StrideDims")) {
        strideDims = iniConfig.getProperty
                                <std::vector<unsigned int> >("StrideDims");
    }
    else if (iniConfig.isProperty("Stride")) {
        strideDims.resize(poolDims.size(),
                             iniConfig.getProperty<unsigned int>("Stride"));
    }
    else {
        strideDims.push_back(iniConfig.getProperty
                                     <unsigned int>("StrideX", 1));
        strideDims.push_back(iniConfig.getProperty
                                     <unsigned int>("StrideY", 1));
        strideDims.push_back(iniConfig.getProperty
                                     <unsigned int>("StrideZ", 1));
        strideDims.resize(poolDims.size(), 1);
    }

    std::vector<unsigned int> paddingDims;

    if (iniConfig.isProperty("PaddingDims")) {
        paddingDims = iniConfig.getProperty
                                <std::vector<unsigned int> >("PaddingDims");
    }
    else if (iniConfig.isProperty("Padding")) {
        paddingDims.resize(poolDims.size(),
                           iniConfig.getProperty<int unsigned>("Padding"));
    }
    else {
        paddingDims.push_back(iniConfig.getProperty
                              <int unsigned>("PaddingX", 0));
        paddingDims.push_back(iniConfig.getProperty
                              <int unsigned>("PaddingY", 0));
        paddingDims.push_back(iniConfig.getProperty
                              <int unsigned>("PaddingZ", 0));
        paddingDims.resize(poolDims.size(), 0);
    }

    const UnpoolCell::Pooling pooling = iniConfig.getProperty
                                      <UnpoolCell::Pooling>("Pooling");

    std::shared_ptr<Activation> activation
        = ActivationGenerator::generate(
            iniConfig, section, model, dataType, "ActivationFunction");

    // Cell construction
    std::shared_ptr<UnpoolCell> cell = Registrar
        <UnpoolCell>::create(model)(network,
                                  section,
                                  poolDims,
                                  nbOutputs,
                                  strideDims,
                                  paddingDims,
                                  pooling,
                                  activation);

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
    MappingGenerator::Mapping defaultMapping
        = MappingGenerator::getMapping(iniConfig, section, "Mapping");

    std::vector<bool> outputConnection(nbOutputs, false);

    for (std::vector<std::shared_ptr<Cell> >::const_iterator it
         = parents.begin(),
         itEnd = parents.end();
         it != itEnd;
         ++it) {
        const Tensor<bool> map = MappingGenerator::generate(
            sp, (*it), nbOutputs, iniConfig, section, "Mapping",
                                                            defaultMapping);

        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int channel = 0; channel < map.dimY(); ++channel) {
                outputConnection[output] = outputConnection[output]
                                            || map(output, channel);
            }
        }

        if (!(*it))
            cell->addInput(sp, 0, 0, sp.getSizeX(), sp.getSizeY(), map);
        else if ((*it)->getOutputsWidth() > 1 || (*it)->getOutputsHeight() > 1)
            cell->addInput((*it).get(), map);
        else {
            throw std::runtime_error("2D input expected for a UnpoolCell (\""
                                     + section + "\"), \"" + (*it)->getName()
                                     + "\" is not,"
                                       " in configuration file: "
                                     + iniConfig.getFileName());
        }
    }

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        if (!outputConnection[output]) {
            std::cout << Utils::cwarning << "Warning: output map #" << output
                      << " of \"" << section << "\" has no input connection."
                      << Utils::cdef << std::endl;
        }
    }

    std::cout << "  # Inputs dims: " << cell->getInputsDims() << std::endl;
    std::cout << "  # Outputs dims: " << cell->getOutputsDims() << std::endl;

    Utils::createDirectories("map");
    cell->writeMap("map/" + section + "_map.dat");

    return cell;
}

void N2D2::UnpoolCellGenerator::postGenerate(const std::shared_ptr<Cell>& cell,
                                             const std::shared_ptr
                                             <DeepNet>& deepNet,
                                             IniParser& iniConfig,
                                             const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const std::string argMax = iniConfig.getProperty<std::string>("ArgMax");
    bool found = false;

    std::shared_ptr<UnpoolCell> unpoolCell
        = std::dynamic_pointer_cast<UnpoolCell>(cell);

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator itCells
         = deepNet->getCells().begin(),
         itCellsEnd = deepNet->getCells().end();
         itCells != itCellsEnd;
         ++itCells)
    {
        if ((*itCells).first == argMax) {
            std::shared_ptr<PoolCell> poolCell = std::dynamic_pointer_cast
                <PoolCell>((*itCells).second);

            if (!poolCell) {
                throw std::runtime_error("Cell name \"" + argMax
                                         + "\" is not a PoolCell for ArgMax ["
                                         + section
                                         + "] in network configuration file: "
                                         + iniConfig.getFileName());
            }

            unpoolCell->addArgMax(poolCell->getArgMax());
            found = true;
            break;
        }
    }

    if (!found) {
        throw std::runtime_error("Cell name \"" + argMax
                                 + "\" not found for ArgMax ["
                                 + section
                                 + "] in network configuration file: "
                                 + iniConfig.getFileName());
    }
}
