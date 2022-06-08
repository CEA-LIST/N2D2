/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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

#include "CMonitor.hpp"
#include "Xnet/Monitor.hpp"
#include "Xnet/Network.hpp"
#include "Xnet/NodeEnv.hpp"
#include "StimuliProvider.hpp"
#include "Xnet/Synapse_PCM.hpp"
#include "Xnet/Synapse_RRAM.hpp"
#include "Xnet/Synapse_Static.hpp"
#include "Cell/Cell_Spike.hpp"
#include "Cell/NodeIn.hpp"
#include "Cell/NodeOut.hpp"
#include "Cell/Cell_CSpike_Top.hpp"
#include "Generator/CellGenerator.hpp"
#include "Generator/CEnvironmentGenerator.hpp"
#include "Generator/DatabaseGenerator.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "Generator/EnvironmentGenerator.hpp"
#include "Generator/TargetGenerator.hpp"

#ifdef ONNX
#include "N2D2.hpp"
#include "Activation/LinearActivation.hpp"
#include "Activation/RectifierActivation.hpp"
#include "Activation/LogisticActivation.hpp"
#include "Activation/SoftplusActivation.hpp"
#include "Database/ILSVRC2012_Database.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "Transformation/PadCropTransformation.hpp"
#include "Transformation/ColorSpaceTransformation.hpp"
#include "Transformation/AffineTransformation.hpp"
#include "Transformation/RangeAffineTransformation.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/ActivationCell.hpp"
#include "Cell/BatchNormCell.hpp"
#include "Cell/ConvCell.hpp"
#include "Cell/DropoutCell.hpp"
#include "Cell/ElemWiseCell.hpp"
#include "Cell/FcCell.hpp"
#include "Cell/LRNCell.hpp"
#include "Cell/PaddingCell.hpp"
#include "Cell/PoolCell.hpp"
#include "Cell/ResizeCell.hpp"
#include "Cell/ReshapeCell.hpp"
#include "Cell/ScalingCell.hpp"
#include "Cell/SoftmaxCell.hpp"
#include "Cell/TransformationCell.hpp"
#include "Generator/ActivationCellGenerator.hpp"
#include "Cell/TransposeCell.hpp"
#include "Generator/BatchNormCellGenerator.hpp"
#include "Generator/ConvCellGenerator.hpp"
#include "Generator/DropoutCellGenerator.hpp"
#include "Generator/FcCellGenerator.hpp"
#include "Generator/ResizeCellGenerator.hpp"
#include "Generator/LRNCellGenerator.hpp"
#include "Generator/PoolCellGenerator.hpp"
#include "Target/TargetCompare.hpp"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#endif

std::shared_ptr<N2D2::DeepNet>
N2D2::DeepNetGenerator::generate(Network& network, const std::string& fileName)
{
    std::string fileExtension = Utils::fileExtension(fileName);
    std::transform(fileExtension.begin(),
                    fileExtension.end(),
                    fileExtension.begin(),
                    ::tolower);

    if (fileExtension == "ini")
        return generateFromINI(network, fileName);
#ifdef ONNX
    else if (fileExtension == "onnx") {
  #ifdef CUDA
        CellGenerator::mDefaultModel = "Frame_CUDA";
  #else
        CellGenerator::mDefaultModel = "Frame";
  #endif

        IniParser dummyParser;
        return generateFromONNX(network, fileName, dummyParser);
    }
#endif
    else {
        throw std::runtime_error(
            "DeepNetGenerator::generate(): unknown file extension: "
            + fileExtension);
    }
}

std::shared_ptr<N2D2::DeepNet>
N2D2::DeepNetGenerator::generateFromINI(Network& network,
                                        const std::string& fileName)
{
    IniParser iniConfig;

    std::cout << "Loading network configuration file " << fileName << std::endl;
    iniConfig.load(fileName);

    // Global parameters
    iniConfig.currentSection();
    CellGenerator::mDefaultModel = iniConfig.getProperty
                                   <std::string>("DefaultModel", "Transcode");
    CellGenerator::mDefaultDataType = iniConfig.getProperty
        <DataType>("DefaultDataType", Float32);

    const bool insertBatchNorm
        = iniConfig.getProperty<bool>("InsertBatchNormAfterConv", false);

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
    deepNet->setParameter("Name", Utils::baseName(fileName));

    if (iniConfig.isSection("database"))
        deepNet->setDatabase(
            DatabaseGenerator::generate(iniConfig, "database"));
    else {
        std::cout << Utils::cwarning << "Warning: no database specified."
                  << Utils::cdef << std::endl;
        deepNet->setDatabase(std::make_shared<Database>());
    }

    // Set up the environment
    bool isEnv = true;

    if (iniConfig.isSection("cenv"))
        deepNet->setStimuliProvider(CEnvironmentGenerator::generate(
            *deepNet->getDatabase(), iniConfig, "cenv"));
    else if (iniConfig.isSection("env"))
        deepNet->setStimuliProvider(EnvironmentGenerator::generate(
            network, *deepNet->getDatabase(), iniConfig, "env"));
    else {
        deepNet->setStimuliProvider(StimuliProviderGenerator::generate(
            *deepNet->getDatabase(), iniConfig, "sp"));
        isEnv = false;
    }

    // Construct network tree
    // std::cout << "Construct network tree..." << std::endl;

    // A map between a INI section and its inputs, e.g. "conv2"->["conv1.1", "conv1.2"]
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

        // Iterate over sections instead of parentLayers to keep INI file order
        for (std::vector<std::string>::const_iterator it = sections.begin(),
             itEnd = sections.end(); it != itEnd; ++it)
        {
            const std::map<std::string, std::vector<std::string> >
                ::const_iterator itParents = parentLayers.find(*it);

            // Skip standalone sections
            if (itParents == parentLayers.end())
                continue;

            unsigned int order = 0;
            bool knownOrder = true;

            // Iterate over all input names of a layer
            for (std::vector<std::string>::const_iterator itParent
                 = (*itParents).second.begin();
                 itParent != (*itParents).second.end();
                 ++itParent)
            {
                const std::vector<std::string>::const_iterator itSections
                    = std::find(sections.begin(), sections.end(), (*itParent));

                // If this parent is not a section, it is assumed that the order
                // is determined by the other parents (this is the case for 
                // ONNX)
                if (itSections != sections.end()) {
                    iniConfig.currentSection(*itSections, false);

                    // If this parent has no "Input" property, we make the same
                    // assumption (probably an ONNX layer for which we added
                    // parameters)
                    if (iniConfig.isProperty("Input")) {
                        const std::map<std::string, unsigned int>
                            ::const_iterator itLayer
                                = layersOrder.find((*itParent));

                        if (itLayer != layersOrder.end())
                            order = std::max(order, (*itLayer).second);
                        else {
                            knownOrder = false;
                            break;
                        }
                    }
                }
            }

            if (knownOrder) {
                layersOrder.insert(std::make_pair((*it), order + 1));

                if (order + 1 >= layers.size())
                    layers.resize(order + 2);

                if (std::find(layers[order + 1].begin(),
                              layers[order + 1].end(),
                              (*it)) == layers[order + 1].end()) {
                    layers[order + 1].push_back((*it));
                    // std::cout << "  " << (*it) << " = " << order + 1 <<
                    // std::endl;

                    ++nbOrderedLayersNext;
                }
            }
        }
    }

    std::set<std::string> ignoreParents;

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        // Iterate over the cell sections of a layer
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it)
        {
            // Set up the layer
            std::vector<std::shared_ptr<Cell> > parentCells;

            for (std::vector<std::string>::const_iterator itParent
                = parentLayers[(*it)].begin();
                itParent != parentLayers[(*it)].end();
                ++itParent)
            {
                if ((*itParent) == "env")
                    parentCells.push_back(std::shared_ptr<Cell>());
                else if (ignoreParents.find((*itParent))
                    == ignoreParents.end())
                {
                    parentCells.push_back(deepNet->getCell((*itParent)));
                }
            }

#ifdef ONNX
            iniConfig.currentSection((*it), false);
            const std::string type = iniConfig.getProperty<std::string>("Type");

            if (type == "ONNX") {
                const std::string fileName
                    = iniConfig.getProperty<std::string>("File");
                std::string fullFileName 
                    = Utils::expandEnvVars(fileName);

                generateFromONNX(network, fullFileName, iniConfig, deepNet,
                                 parentCells);

                const std::vector<std::string> targets
                    = iniConfig.getSections("*.Target*");

                for (std::vector<std::string>::const_iterator itTarget
                    = targets.begin(),
                    itTargetEnd = targets.end();
                    itTarget != itTargetEnd;
                    ++itTarget)
                {
                    std::size_t targetPos = (*itTarget).find(".Target");
                    const std::string cellName
                        = (*itTarget).substr(0, targetPos);

                    if (deepNet->hasCell(cellName)) {
                        std::shared_ptr<Cell> cell = deepNet->getCell(cellName);
                        std::shared_ptr<Target> target
                            = TargetGenerator::generate(
                                cell, deepNet, iniConfig, (*itTarget));
                        deepNet->addTarget(target);
                    }
                }

                ignoreParents.insert((*it));
            } // Else set up from INI section
            else {
#endif
                std::shared_ptr<Cell> cell
                    = CellGenerator::generate(network, *deepNet,
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

                std::shared_ptr<Cell_CSpike_Top> cellCSpike
                    = std::dynamic_pointer_cast<Cell_CSpike_Top>(cell);
                // Monitor for the cell
                // Try different casts to find out Cell type
                std::shared_ptr<Cell_Spike> cellSpike
                    = std::dynamic_pointer_cast <Cell_Spike>(cell);

                if (cellCSpike) {
                    std::shared_ptr<CMonitor> monitor;
    #ifdef CUDA
                    if (cellCSpike->isCuda()) {
                        monitor = std::make_shared<CMonitor_CUDA>();
                    }
                    else {
    #endif
                        monitor = std::make_shared<CMonitor>();
    #ifdef CUDA
                    }
    #endif
                    monitor->add(cellCSpike->getOutputs());
                    deepNet->addCMonitor((*it), monitor);
                }
                else if (cellSpike) {
                    std::shared_ptr<Monitor> monitor(new Monitor(network));
                    monitor->add(cellSpike->getOutputs());

                    deepNet->addMonitor((*it), monitor);

                }
                else if (isEnv) {
                    // Don't warn if we are using a StimuliProvider
                    std::cout << "Warning: No monitor could be added to Cell: "
                        << cell->getName() << std::endl;
                }
#ifdef ONNX
            }
#endif
        }
    }

    deepNet->removeExtraReshape();

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

    std::shared_ptr<Environment> env = std::dynamic_pointer_cast<Environment>
        (deepNet->getStimuliProvider());

    std::shared_ptr<CEnvironment> Cenv = std::dynamic_pointer_cast<CEnvironment>
        (deepNet->getStimuliProvider());

    if (Cenv) {
#ifdef CUDA
        std::shared_ptr<CMonitor> cmonitor(new CMonitor_CUDA());
        cmonitor->add(Cenv->getTickData());

        deepNet->addCMonitor("env", cmonitor);

#else
        std::shared_ptr<CMonitor> cmonitor(new CMonitor());
        cmonitor->add(Cenv->getTickData());

        deepNet->addCMonitor("env", cmonitor);

#endif
    }

    else if (env) {
        std::shared_ptr<Monitor> monitor(new Monitor(network));
        monitor->add(env->getNodes());

        deepNet->addMonitor("env", monitor);
    }
    else {
	 std::runtime_error(
	"DeepNetGenerator::generate: Cast of environment failed. No Monitors added");
    }


    // Check that the properties of the latest section are valid
    iniConfig.currentSection();

    if (insertBatchNorm)
        deepNet->insertBatchNormAfterConv();


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

#ifdef ONNX
std::shared_ptr<N2D2::DeepNet>
N2D2::DeepNetGenerator::generateFromONNX(Network& network,
    const std::string& fileName,
    IniParser& iniConfig,
    std::shared_ptr<DeepNet> deepNet,
    const std::vector<std::shared_ptr<Cell> >& parentCells)
{
    if (!deepNet) {
        deepNet = std::shared_ptr<DeepNet>(new DeepNet(network));
        deepNet->setParameter("Name", Utils::baseName(fileName));
    }

    if (!deepNet->getDatabase()) {
        std::cout << Utils::cwarning << "Warning: no database specified."
                    << Utils::cdef << std::endl;
        deepNet->setDatabase(std::make_shared<Database>());
        //std::shared_ptr<ILSVRC2012_Database> database = std::make_shared
        //    <ILSVRC2012_Database>(1.0, true, true);
        //database->load(N2D2_DATA("ILSVRC2012"),
        //            N2D2_DATA("ILSVRC2012/synsets.txt"));
        //deepNet->setDatabase(database);
    }

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    onnx::ModelProto onnxModel;

    std::ifstream onnxFile(fileName.c_str(), std::ios::binary);
    if (!onnxFile.good())
        throw std::runtime_error("Could not open ONNX file: " + fileName);
    google::protobuf::io::IstreamInputStream zero_copy_input(&onnxFile);
    google::protobuf::io::CodedInputStream coded_input(&zero_copy_input);

#if GOOGLE_PROTOBUF_VERSION < 3006000
    coded_input.SetTotalBytesLimit(1073741824, 536870912);
#else
    coded_input.SetTotalBytesLimit(1073741824);
#endif

    
    if (!onnxModel.ParseFromCodedStream(&coded_input)
        || !onnxFile.eof())
    {
        throw std::runtime_error("Failed to parse ONNX file: " + fileName);
    }

    onnxFile.close();

    std::cout << "Importing ONNX model:\n"
        "  ir_version = " << onnxModel.ir_version() << "\n"
        "  producer_name = " << onnxModel.producer_name() << "\n"
        "  producer_version = " << onnxModel.producer_version() << "\n"
        "  domain = " << onnxModel.domain() << "\n"
        "  model_version = " << onnxModel.model_version() << "\n"
        "  doc_string = " << onnxModel.doc_string() << std::endl;

    int opsetVersion = -1;

    for (int i = 0; i < onnxModel.opset_import_size(); ++i) {
        onnx::OperatorSetIdProto opset = onnxModel.opset_import(i);

        if (opset.domain() == "")
            opsetVersion = opset.version();
    }

    std::cout << "Opset version is: " << opsetVersion << std::endl;

    ONNX_processGraph(deepNet, parentCells,
                      onnxModel.graph(), opsetVersion, iniConfig);

    // TF exported ONNX can create extra transpose layers
    deepNet->removeExtraTranspose();
    deepNet->removeExtraReshape();

    return deepNet;
}

std::shared_ptr<N2D2::BaseTensor> N2D2::DeepNetGenerator::ONNX_unpackTensor(
    const onnx::TensorProto* onnxTensor,
    const std::vector<unsigned int>& expectedDims)
{
    const std::string dataTypeName = onnxTensor->GetTypeName();

    if (onnxTensor->data_type() == onnx::TensorProto_DataType_FLOAT) {
        return std::make_shared<Tensor<float> >(
            ONNX_unpackTensor<float>(onnxTensor, expectedDims));
    }
    else if (onnxTensor->data_type() == onnx::TensorProto_DataType_FLOAT16) {
        return std::make_shared<Tensor<half_float::half> >(
            ONNX_unpackTensor<half_float::half>(onnxTensor, expectedDims));
    }
    else if (onnxTensor->data_type() == onnx::TensorProto_DataType_DOUBLE) {
        return std::make_shared<Tensor<double> >(
            ONNX_unpackTensor<double>(onnxTensor, expectedDims));
    }
    else if (onnxTensor->data_type() == onnx::TensorProto_DataType_INT8) {
        return std::make_shared<Tensor<int8_t> >(
            ONNX_unpackTensor<int8_t>(onnxTensor, expectedDims));
    }
    else if (onnxTensor->data_type() == onnx::TensorProto_DataType_INT16) {
        return std::make_shared<Tensor<int16_t> >(
            ONNX_unpackTensor<int16_t>(onnxTensor, expectedDims));
    }
    else if (onnxTensor->data_type() == onnx::TensorProto_DataType_INT32) {
        return std::make_shared<Tensor<int32_t> >(
            ONNX_unpackTensor<int32_t>(onnxTensor, expectedDims));
    }
    else if (onnxTensor->data_type() == onnx::TensorProto_DataType_INT64) {
        return std::make_shared<Tensor<int64_t> >(
            ONNX_unpackTensor<int64_t>(onnxTensor, expectedDims));
    }
    else if (onnxTensor->data_type() == onnx::TensorProto_DataType_UINT64) {
        return std::make_shared<Tensor<uint64_t> >(
            ONNX_unpackTensor<uint64_t>(onnxTensor, expectedDims));
    }
    else {
        std::ostringstream errorStr;
        errorStr << "Unsupported type for ONNX tensor \""
            << onnxTensor->name() << "\": " << dataTypeName << std::endl;

        throw std::runtime_error(errorStr.str());
    }
}

void N2D2::DeepNetGenerator::ONNX_processGraph(
    std::shared_ptr<DeepNet> deepNet,
    const std::vector<std::shared_ptr<Cell> >& graphParentCells,
    const onnx::GraphProto& graph,
    int opsetVersion,
    IniParser& iniConfig)
{
    const std::string onnxName = iniConfig.getCurrentSection();
    const std::string model = CellGenerator::mDefaultModel;

    std::map<std::string, const onnx::TensorProto*> initializer;
    for (int i = 0; i < graph.initializer_size(); ++i) {
        const onnx::TensorProto* tensor = &(graph.initializer(i));
        initializer[tensor->name()] = tensor;
    }
    std::map<std::string, std::vector<size_t> > shape;

    const bool globTranspose
        = iniConfig.getProperty<bool>("Transpose", false);

    const bool globCNTK
        = iniConfig.getProperty<bool>("CNTK", false);

    // Map the ONNX graph inputs to the graphParentCells
    std::map<std::string, std::shared_ptr<Cell> > inputsMapping;
    std::shared_ptr<StimuliProvider> sp;
    unsigned int nbInputs = 0;

    for (int i = 0; i < graph.input_size(); ++i) {
        const onnx::ValueInfoProto* valueInfo = &(graph.input(i));

        // If there is a constant initializer, no parent is required
        if (initializer.find(valueInfo->name()) != initializer.end())
            continue;

        // Not enough parents in graphParentCells
        if (nbInputs >= graphParentCells.size()) {
            std::stringstream msgStr;
            msgStr << "The number of parents provided ("
                << graphParentCells.size() << ") is less than the required "
                "number of data input in the ONNX graph";
            throw std::runtime_error(msgStr.str());
        }

        std::shared_ptr<Cell> parentCell = graphParentCells[nbInputs];
        ++nbInputs;

        inputsMapping[valueInfo->name()] = parentCell;

        if (parentCell)
            continue;

        // parentCell is the StimuliProvider
        const onnx::TypeProto_Tensor& inputType
            = valueInfo->type().tensor_type();
        const onnx::TensorShapeProto& inputShape
            = inputType.shape();

        std::vector<size_t> size;
        for (int i = 1; i < inputShape.dim_size(); ++i)
            size.push_back(inputShape.dim(i).dim_value());
        std::reverse(size.begin(), size.end());

        if (globTranspose && size.size() >= 2)
            std::swap(size[0], size[1]);

        if (!deepNet->getStimuliProvider()) {
            // Input: StimuliProvider construction
            unsigned int batchSize = inputShape.dim(0).dim_value();
            if (batchSize < 1)
                batchSize = 1;

            const bool compositeStimuli = false;

            sp = std::shared_ptr<StimuliProvider>(new StimuliProvider(
                *deepNet->getDatabase(), size, batchSize, compositeStimuli));
            deepNet->setStimuliProvider(sp);

            std::cout << "StimuliProvider: " << size << " (" << batchSize << ")"
                << std::endl;
        }
        else {
            sp = deepNet->getStimuliProvider();

            const bool ignoreInputSize
                = iniConfig.getProperty<bool>("IgnoreInputSize", false);

            if (!ignoreInputSize && sp->getSize() != size
                && !(std::equal(size.begin(), size.end(), sp->getSize().begin())
                    && std::all_of(sp->getSize().begin() + size.size(),
                                sp->getSize().end(), [](size_t i){return i == 1;})))
            {
                std::cout << Utils::cwarning << "Unexpected size for ONNX input \""
                    << valueInfo->name() << "\": got " << size
                    << " , but StimuliProvider provides " << sp->getSize()
                    << Utils::cdef << std::endl;
            }
        }
    }
    const bool initializeFromONNX
            = iniConfig.getProperty<bool>("ONNX_init", true);

    if (nbInputs != graphParentCells.size()) {
        std::stringstream msgStr;
        msgStr << "The number of parents provided (" << graphParentCells.size()
            << ") does not match the number of data input in the ONNX graph ("
            << nbInputs << ")";
        throw std::runtime_error(msgStr.str());
    }

    auto getCell = [&inputsMapping, &deepNet](const std::string& name)
    {
        auto it = inputsMapping.find(name);
        return (it != inputsMapping.end())
            ? (*it).second  // Input
            : deepNet->getCell(name);
    };

    // Cells
    std::shared_ptr<Cell> cell;
    std::map<std::string, std::vector<std::string> > concat;
    std::map<std::string, std::string> redirect;

    auto redirectName = [&redirect](const std::string& name)
    {
        const std::map<std::string, std::string>::const_iterator it
            = redirect.find(name);
        return (it != redirect.end()) ? (*it).second : name;
    };

    const std::vector<std::string> ignore
        = iniConfig.getProperty<std::vector<std::string> >("Ignore",
            std::vector<std::string>());

    for (int n = 0; n < graph.node_size(); ++n) {
        const onnx::NodeProto& node = graph.node(n);

        std::cout << "Layer: " << node.output(0) << " [" << node.op_type()
            << "]" << std::endl;
/*
        // DEBUG
        std::cout << "  Input(s): ";
        for (int i = 0; i < node.input_size(); ++i)
            std::cout << redirectName(node.input(i)) << " ";
        std::cout << std::endl;

        std::cout << "  Output(s): ";
        for (int i = 0; i < node.output_size(); ++i)
            std::cout << node.output(i) << " ";
        std::cout << std::endl;
*/
        std::map<std::string, const onnx::AttributeProto*> attribute;
        for (int a = 0; a < node.attribute_size(); ++a) {
            const onnx::AttributeProto* attr = &(node.attribute(a));
            attribute[attr->name()] = attr;
        }

        std::map<std::string, const onnx::AttributeProto*>
            ::const_iterator itAttr;
        std::map<std::string, const onnx::TensorProto*>::const_iterator itInit;
        std::map<std::string, std::vector<size_t> >::const_iterator itShape;

        if (std::find(ignore.begin(), ignore.end(), node.output(0))
            != ignore.end())
        {
            std::cout << "  Ignore " << node.output(0) << " layer as requested."
                << std::endl;
            continue;
        }

        //Abs
        //Acos
        //Acosh
        //Add -> see Sum
        //And
        //ArgMax
        //ArgMin
        //Asin
        //Asinh
        //Atan
        //Atanh
        if (node.op_type() == "AveragePool"
            || node.op_type() == "GlobalAveragePool"
            || node.op_type() == "ReduceMean"
            || node.op_type() == "MaxPool"
            || node.op_type() == "GlobalMaxPool"
            || node.op_type() == "ReduceMax")
        {
            const std::string inputX = redirectName(node.input(0));
            unsigned int nbOutputs = 0;
            std::vector<size_t> inputsDims;

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                    const std::string input = (*itConcat).second[i];
                    std::shared_ptr<Cell> inputCell = getCell(input);

                    nbOutputs += inputCell->getNbOutputs();
                    inputsDims = inputCell->getOutputsDims();
                }
            }
            else {
                std::shared_ptr<Cell> inputXCell = getCell(inputX);
                
                if (inputXCell) {
                    nbOutputs += inputXCell->getNbOutputs();
                    inputsDims = inputXCell->getOutputsDims();
                }
            }

            // kernel_shape
            std::vector<unsigned int> kernelDims;

            if ((itAttr = attribute.find("kernel_shape")) != attribute.end()) {
                for (int dim = 0; dim < (*itAttr).second->ints_size(); ++dim)
                    kernelDims.push_back((*itAttr).second->ints(dim));

                std::reverse(kernelDims.begin(), kernelDims.end());
            }
            else if (node.op_type() == "GlobalAveragePool"
                || node.op_type() == "ReduceMean"
                || node.op_type() == "GlobalMaxPool"
                || node.op_type() == "ReduceMax")
            {
                assert(!inputsDims.empty());
                kernelDims = std::vector<unsigned int>(inputsDims.begin(),
                                                       inputsDims.end());
                kernelDims.pop_back();  // remove number of channels
            }

            // strides
            std::vector<unsigned int> strideDims;

            if ((itAttr = attribute.find("strides")) != attribute.end()) {
                for (int dim = 0; dim < (*itAttr).second->ints_size(); ++dim)
                    strideDims.push_back((*itAttr).second->ints(dim));

                std::reverse(strideDims.begin(), strideDims.end());
            }
            else
                strideDims.resize(kernelDims.size(), 1);

            // pads
            std::vector<unsigned int> paddingDimsBegin;
            std::vector<unsigned int> paddingDimsEnd;

            if ((itAttr = attribute.find("pads")) != attribute.end()) {
                // Mutually exclusive with auto_pad
                assert((*itAttr).second->ints_size() % 2 == 0);
                const int offset = (*itAttr).second->ints_size() / 2;

                for (int dim = 0; dim < offset; ++dim) {
                    paddingDimsBegin.push_back((*itAttr).second->ints(dim));
                    paddingDimsEnd.push_back((*itAttr).second->ints(offset + dim));
                }
            }
            else if ((itAttr = attribute.find("auto_pad")) != attribute.end()) {
                // Mutually exclusive with pads
                for (unsigned int dim = 0; dim < kernelDims.size(); ++dim) {
                    const int padding = (kernelDims[dim] - strideDims[dim]);
                    const int floorHalfPadding = (padding / 2);

                    if ((*itAttr).second->s() == "SAME_UPPER") {
                        paddingDimsBegin.push_back(floorHalfPadding);
                        paddingDimsEnd.push_back(padding - floorHalfPadding);
                    }
                    else if ((*itAttr).second->s() == "SAME_LOWER") {
                        paddingDimsBegin.push_back(padding - floorHalfPadding);
                        paddingDimsEnd.push_back(floorHalfPadding);
                    }
                    else if ((*itAttr).second->s() == "VALID") {
                        paddingDimsBegin.push_back(0);
                        paddingDimsEnd.push_back(0);
                    }
                }
            }
            else {
                paddingDimsBegin.resize(kernelDims.size(), 0);
                paddingDimsEnd.resize(kernelDims.size(), 0);
            }

            std::reverse(paddingDimsBegin.begin(), paddingDimsBegin.end());
            std::reverse(paddingDimsEnd.begin(), paddingDimsEnd.end());

            if ((itAttr = attribute.find("ceil_mode")) != attribute.end()) {
                if ((*itAttr).second->i() == 1) {
                    const int inputX = inputsDims[0];
                    const int inputY = inputsDims[1];
                    const int kX = kernelDims[0];
                    const int kY = kernelDims[1];
                    const int pX = paddingDimsBegin[0];
                    const int pY = paddingDimsBegin[1];
                    const int sX = strideDims[0];
                    const int sY = strideDims[1];

                    const int outputXCeil 
                        = std::ceil( (float)(inputX - kX + 2 * pX) / (float) sX ) + 1;
                    const int outputYCeil 
                        = std::ceil( (float)(inputY - kY + 2 * pY) / (float)sY ) + 1;
                    const int outputXFloor 
                        = std::floor( (float)(inputX - kX + 2 * pX) / (float) sX ) + 1;
                    const int outputYFloor  
                        = std::floor( (float)(inputY - kY + 2 * pY) / (float)sY ) + 1;

                    if(outputXCeil > outputXFloor) {
                        paddingDimsEnd[0] += (outputXCeil - outputXFloor);
                    }
                    if(outputYCeil > outputYFloor) {
                        paddingDimsEnd[1] += (outputYCeil - outputYFloor);
                    }
                }
            }

            if ((itAttr = attribute.find("count_include_pad"))
                != attribute.end())
            {
                if ((*itAttr).second->i() != 0) {
                    std::cout << Utils::cwarning << "Unsupported operation: "
                        << node.op_type() << " with count_include_pad != 0"
                        << Utils::cdef << std::endl;
                }
            }

            std::shared_ptr<Activation> activation
                = std::shared_ptr<Activation>();

            // Asymmetric padding
            bool paddingCellRequired = false;

            for (unsigned int dim = 0; dim < paddingDimsBegin.size(); ++dim) {
                if (paddingDimsBegin[dim] != paddingDimsEnd[dim]) {
                    paddingCellRequired = true;
                    break;
                }
            }

            std::vector<unsigned int> paddingDims = (paddingCellRequired)
                ? std::vector<unsigned int>(kernelDims.size(), 0U)
                : paddingDimsBegin;

            // Make a unit map
            Tensor<bool> map({nbOutputs,
                              nbOutputs}, false);

            for (unsigned int i = 0; i < nbOutputs; ++i)
                map(i, i) = true;

            const PoolCell::Pooling pooling = (node.op_type() == "AveragePool"
                || node.op_type() == "GlobalAveragePool"
                || node.op_type() == "ReduceMean")
                    ? PoolCell::Average : PoolCell::Max;

            if (globTranspose) {
                if (kernelDims.size() < 2) {
                    kernelDims.resize(2, 1);
                    strideDims.resize(2, 1);
                    paddingDims.resize(2, 0);
                }

                std::swap(kernelDims[0], kernelDims[1]);
                std::swap(strideDims[0], strideDims[1]);
                std::swap(paddingDims[0], paddingDims[1]);
            }

            std::shared_ptr<PoolCell> poolCell
                = Registrar<PoolCell>::create<Float_T>(model)(deepNet->getNetwork(),
                                                                *deepNet, 
                                                                node.output(0),
                                                                kernelDims,
                                                                nbOutputs,
                                                                strideDims,
                                                                paddingDims,
                                                                pooling,
                                                                activation);

            if (iniConfig.currentSection(node.output(0), false)) {
                PoolCellGenerator::generateParams(poolCell, iniConfig,
                    node.output(0), model, Float32);
            }
            else if (iniConfig.currentSection(onnxName + ":Pool_def", false)) {
                PoolCellGenerator::generateParams(poolCell, iniConfig,
                    onnxName + ":Pool_def", model, Float32);
            }

            std::vector<std::shared_ptr<Cell> > parentCells;

            if (paddingCellRequired) {
                std::cout << "  Added padding: " << paddingDimsBegin
                    << " -- " << paddingDimsEnd << std::endl;

                if (globTranspose) {
                    std::swap(paddingDimsBegin[0], paddingDimsBegin[1]);
                    std::swap(paddingDimsEnd[0], paddingDimsEnd[1]);
                }

                std::shared_ptr<PaddingCell> paddingCell = Registrar
                    <PaddingCell>::create(model)(*deepNet,
                                                node.output(0) + "_padding",
                                                nbOutputs,
                                                paddingDimsBegin[1],
                                                paddingDimsEnd[1],
                                                paddingDimsBegin[0],
                                                paddingDimsEnd[0]);

                if ((itConcat = concat.find(inputX)) != concat.end()) {
                    for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                        const std::string input = (*itConcat).second[i];
                        std::shared_ptr<Cell> inputCell = getCell(input);
                        parentCells.push_back(inputCell);

                        paddingCell->addInput(inputCell.get());
                    }
                }
                else {
                    std::shared_ptr<Cell> inputXCell = getCell(inputX);
                    parentCells.push_back(inputXCell);

                    if (inputXCell)
                        paddingCell->addInput(inputXCell.get());
                    else {
                        paddingCell->addInput(*sp, 0, 0,
                                              sp->getSizeX(), sp->getSizeY());
                    }
                }

                poolCell->addInput(paddingCell.get(), map);

                deepNet->addCell(paddingCell, parentCells);
                paddingCell->initialize();

                parentCells.clear();
                parentCells.push_back(paddingCell);
            }
            else {
                if ((itConcat = concat.find(inputX)) != concat.end()) {
                    unsigned int mapOffset = 0;

                    for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                        const std::string input = (*itConcat).second[i];
                        std::shared_ptr<Cell> inputCell = getCell(input);
                        parentCells.push_back(inputCell);

                        // Make a unit map
                        Tensor<bool> inputMap({nbOutputs,
                                               inputCell->getNbOutputs()}, false);

                        for (unsigned int i = 0; i < inputCell->getNbOutputs();
                            ++i)
                        {
                            inputMap(mapOffset + i, i) = true;
                        }

                        poolCell->addInput(inputCell.get(), inputMap);
                        mapOffset += inputCell->getNbOutputs();
                    }
                }
                else {
                    std::shared_ptr<Cell> inputXCell = getCell(inputX);
                    parentCells.push_back(inputXCell);

                    if (inputXCell)
                        poolCell->addInput(inputXCell.get(), map);
                    else {
                        poolCell->addInput(*sp, 0, 0,
                                           sp->getSizeX(), sp->getSizeY(), map);
                    }
                }
            }

            deepNet->addCell(poolCell, parentCells);
            poolCell->initialize();
            cell = poolCell;

            poolCell->writeMap("map/" + node.output(0) + "_map.dat");
/*
            // DEBUG
            std::string targetName = Utils::dirName(node.output(0));
            targetName.pop_back();
            targetName = Utils::baseName(targetName);

            if (!targetName.empty()) {
                std::shared_ptr<Target> target = Registrar
                    <Target>::create("TargetCompare")(node.output(0)
                                                                    + ".Target",
                                                poolCell,
                                                deepNet->getStimuliProvider(),
                                                1.0,
                                                0.0,
                                                1,
                                                "",
                                                false);
                target->setParameter<std::string>("DataPath",
                                                  "n07745940_14257");
                target->setParameter<std::string>("Matching",
                                                  targetName + ".txt");
                target->setParameter<TargetCompare::TargetFormat>(
                    "TargetFormat", TargetCompare::NHWC);
                target->setParameter<bool>("LogError", true);
                target->setParameter<unsigned int>("BatchPacked", 1);

                deepNet->addTarget(target);
            }
*/
        }
        else if (node.op_type() == "BatchNormalization") {
            const std::string inputScale = node.input(1);
            const unsigned int nbOutputs = initializer[inputScale]->dims(0);

            std::shared_ptr<Activation> activation
                = Registrar<LinearActivation>::create<Float_T>(model)();

            std::shared_ptr<BatchNormCell> batchNormCell
                = Registrar<BatchNormCell>::create<Float_T>(model)(*deepNet, 
                                                                node.output(0),
                                                                nbOutputs,
                                                                activation);

            // Parameters
            float epsilon = 1.0e-5;
            float momentum = 0.9;

            if ((itAttr = attribute.find("epsilon")) != attribute.end())
                epsilon = (*itAttr).second->f();

            if ((itAttr = attribute.find("momentum")) != attribute.end())
                momentum = (*itAttr).second->f();

            batchNormCell->setParameter<double>("Epsilon", epsilon);
            batchNormCell->setParameter<double>("MovingAverageMomentum",
                                                momentum);

            if (iniConfig.currentSection(node.output(0), false)) {
                BatchNormCellGenerator::generateParams(batchNormCell, iniConfig,
                    node.output(0), model, Float32);
            }
            else if (iniConfig.currentSection(onnxName + ":BatchNorm_def", false)) {
                BatchNormCellGenerator::generateParams(batchNormCell, iniConfig,
                    onnxName + ":BatchNorm_def", model, Float32);
            }

            const std::string inputX = redirectName(node.input(0));
            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                throw std::runtime_error("Unsupported operation: Concat before "
                    "BatchNorm");
            }
            else {
                std::shared_ptr<Cell> inputXCell = getCell(inputX);
                parentCells.push_back(inputXCell);

                if (inputXCell)
                    batchNormCell->addInput(inputXCell.get());
                else {
                    batchNormCell->addInput(*sp, 0, 0,
                                        sp->getSizeX(), sp->getSizeY());
                }
            }

            deepNet->addCell(batchNormCell, parentCells);
            batchNormCell->initialize();
            cell = batchNormCell;
/*
            // DEBUG
            std::string targetName = Utils::dirName(node.output(0));
            targetName.pop_back();
            targetName = Utils::dirName(targetName);
            targetName.pop_back();
            targetName = Utils::baseName(targetName);

            if (!targetName.empty()) {
                std::shared_ptr<Target> target = Registrar
                    <Target>::create("TargetCompare")(node.output(0)
                                                                    + ".Target",
                                                batchNormCell,
                                                deepNet->getStimuliProvider(),
                                                1.0,
                                                0.0,
                                                1,
                                                "",
                                                false);
                target->setParameter<std::string>("DataPath",
                                                  "n07745940_14257");
                target->setParameter<std::string>("Matching",
                                                  targetName + ".txt");
                target->setParameter<TargetCompare::TargetFormat>(
                    "TargetFormat", TargetCompare::NHWC);
                target->setParameter<bool>("LogError", true);
                target->setParameter<unsigned int>("BatchPacked", 1);

                deepNet->addTarget(target);
            }
*/
            if(initializeFromONNX) {
            // Free parameters
                if ((itInit = initializer.find(node.input(1))) != initializer.end())
                {
                    Tensor<Float_T> scale
                        = ONNX_unpackTensor<Float_T>((*itInit).second,
                            {(unsigned int)batchNormCell->getNbOutputs()});
                    scale.reshape({1, batchNormCell->getNbOutputs()});

                    for (unsigned int output = 0;
                        output < batchNormCell->getNbOutputs(); ++output)
                    {
                        batchNormCell->setScale(output, scale[output]);
                    }
                }
                else {
                    std::cout << "  No initializer for \"" << node.input(1)
                        << "\"" << std::endl;
                }

                if ((itInit = initializer.find(node.input(2))) != initializer.end())
                {
                    Tensor<Float_T> bias
                        = ONNX_unpackTensor<Float_T>((*itInit).second,
                            {(unsigned int)batchNormCell->getNbOutputs()});
                    bias.reshape({1, batchNormCell->getNbOutputs()});

                    for (unsigned int output = 0;
                        output < batchNormCell->getNbOutputs(); ++output)
                    {
                        batchNormCell->setBias(output, bias[output]);
                    }
                }
                else {
                    std::cout << "  No initializer for \"" << node.input(2)
                        << "\"" << std::endl;
                }

                if ((itInit = initializer.find(node.input(3))) != initializer.end())
                {
                    Tensor<Float_T> mean
                        = ONNX_unpackTensor<Float_T>((*itInit).second,
                            {(unsigned int)batchNormCell->getNbOutputs()});
                    mean.reshape({1, batchNormCell->getNbOutputs()});

                    for (unsigned int output = 0;
                        output < batchNormCell->getNbOutputs(); ++output)
                    {
                        batchNormCell->setMean(output, mean[output]);
                    }
                }
                else {
                    std::cout << "  No initializer for \"" << node.input(3)
                        << "\"" << std::endl;
                }

                if ((itInit = initializer.find(node.input(4))) != initializer.end())
                {
                    Tensor<Float_T> variance
                        = ONNX_unpackTensor<Float_T>((*itInit).second,
                            {(unsigned int)batchNormCell->getNbOutputs()});
                    variance.reshape({1, batchNormCell->getNbOutputs()});

                    for (unsigned int output = 0;
                        output < batchNormCell->getNbOutputs(); ++output)
                    {
                        batchNormCell->setVariance(output, variance[output]);
                    }
                }
                else {
                    std::cout << "  No initializer for \"" << node.input(4)
                        << "\"" << std::endl;
                }
            }
        }
        //BitShift
        else if (node.op_type() == "Cast") {
            std::cout << Utils::cnotice << "  Ignore Cast operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
/*
            std::cout <<  "  Ignore Cast operation to "
                << node.attribute(0).GetTypeName()
                <<  std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            assert(node.attribute_size() > 0);

            continue;
            */
        }
        //Ceil
        else if (node.op_type() == "Clip") {
            Float_T minVal = std::numeric_limits<Float_T>::lowest();
            Float_T maxVal = std::numeric_limits<Float_T>::max();

            if (node.input_size() > 1) {
                const std::string inputMin = redirectName(node.input(1));

                if ((itInit = initializer.find(inputMin))
                    != initializer.end())
                {
                    const Tensor<Float_T> constantMin
                        = tensor_cast<Float_T>(*ONNX_unpackTensor((*itInit).second));

                    if (constantMin.size() == 1)
                        minVal = constantMin(0);
                    else {
                        std::ostringstream errorStr;
                        errorStr << "Unsupported ONNX operator: " << node.op_type()
                            << " with input min. of dim != 1" << std::endl;

                        throw std::runtime_error(errorStr.str());
                    }
                }
                else {
                    std::ostringstream errorStr;
                    errorStr << "Unsupported ONNX operator: " << node.op_type()
                        << " with non-const input min." << std::endl;

                    throw std::runtime_error(errorStr.str());
                }
            }
            else {
                if ((itAttr = attribute.find("min")) != attribute.end())
                    minVal = (*itAttr).second->f();
            }

            if (node.input_size() > 2) {
                const std::string inputMax = redirectName(node.input(2));

                if ((itInit = initializer.find(inputMax))
                    != initializer.end())
                {
                    const Tensor<Float_T> constantMax
                        = tensor_cast<Float_T>(*ONNX_unpackTensor((*itInit).second));

                    if (constantMax.size() == 1)
                        maxVal = constantMax(0);
                    else {
                        std::ostringstream errorStr;
                        errorStr << "Unsupported ONNX operator: " << node.op_type()
                            << " with input max. of dim != 1" << std::endl;

                        throw std::runtime_error(errorStr.str());
                    }
                }
                else {
                    std::ostringstream errorStr;
                    errorStr << "Unsupported ONNX operator: " << node.op_type()
                        << " with non-const input max." << std::endl;

                    throw std::runtime_error(errorStr.str());
                }
            }
            else {
                if ((itAttr = attribute.find("max")) != attribute.end())
                    maxVal = (*itAttr).second->f();
            }

            if (minVal == 0.0 && maxVal > 0.0) {
                std::shared_ptr<Activation> activation
                    = Registrar<RectifierActivation>::create<Float_T>(model)();

                if (maxVal != std::numeric_limits<Float_T>::max())
                    activation->setParameter<double>("Clipping", maxVal);

                std::shared_ptr<Cell_Frame_Top> cellFrame
                    = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

                if (cellFrame->getActivation()
                    && cellFrame->getActivation()->getType()
                        != LinearActivation::Type)
                {
                    if (cellFrame->getActivation()->getType()
                        == RectifierActivation::Type)
                    {
                        const double oldClipping = cellFrame->getActivation()
                            ->getParameter<double>("Clipping");

                        if (oldClipping == 0.0 || oldClipping > maxVal)
                            cellFrame->setActivation(activation);
                    }
                    else {
                        throw std::runtime_error("Cell " + cell->getName()
                            + " already has an activation!");
                    }
                }
                else
                    cellFrame->setActivation(activation);
                std::cout << "  clipping in [" << minVal << ", " << maxVal << "]"
                    << std::endl;
            }
            else if (minVal == std::numeric_limits<Float_T>::lowest() && maxVal > 0.0){
                std::shared_ptr<Activation> activation
                    = Registrar<RectifierActivation>::create<Float_T>(model)();

                if (maxVal != std::numeric_limits<Float_T>::max())
                    activation->setParameter<double>("Clipping", maxVal);

                std::shared_ptr<Cell_Frame_Top> cellFrame
                    = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

                if (cellFrame->getActivation()
                    && cellFrame->getActivation()->getType()
                        != LinearActivation::Type)
                {
                    if (cellFrame->getActivation()->getType()
                        == RectifierActivation::Type)
                    {
                        const double oldClipping = cellFrame->getActivation()
                            ->getParameter<double>("Clipping");

                        if (oldClipping == 0.0 || oldClipping > maxVal){
                            cellFrame->setActivation(activation);
                            minVal = 0.0;
                            std::cout << "  clipped ReLu will be set instead of ReLu"  << std::endl;
                        }
                    }
                    else {
                        throw std::runtime_error("Cell " + cell->getName()
                            + " already has an activation!");
                    }
                }
                else
                    cellFrame->setActivation(activation);
            }
            else if (minVal < 0.0 && maxVal > 0.0) {
                std::shared_ptr<Cell_Frame_Top> cellFrame
                    = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
                if (!cellFrame->getActivation()) {
                    std::shared_ptr<Activation> activation
                        = Registrar<LinearActivation>::create<Float_T>(model)();
                    cellFrame->setActivation(activation);
                }
                else if (cellFrame->getActivation()->getType()
                        != LinearActivation::Type) {
                    std::ostringstream errorStr;
                    errorStr << "Unsupported ONNX operator: " << node.op_type()
                        << " with min=" << minVal << " and max=" << maxVal
                        << " when non linear activation" << std::endl;

                    throw std::runtime_error(errorStr.str());
                }
                
                std::cout << Utils::cnotice << "  Ignore Clip operation when "
                            << " min is inferior to 0 and max is superior to 0"
                            << Utils::cdef << std::endl;
            }
            else {
                std::ostringstream errorStr;
                errorStr << "Unsupported ONNX operator: " << node.op_type()
                    << " with min=" << minVal << " and max=" << maxVal
                    << std::endl;

                throw std::runtime_error(errorStr.str());
            }

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //Compress
        else if (node.op_type() == "Concat") {
            //IK: create concat cell, represented by activation cell with linear activation
            /*
            bool newInsert;
            std::map<std::string, std::vector<std::string> >::iterator it;
            std::tie(it, newInsert) = concat.insert(
                std::make_pair(node.output(0), std::vector<std::string>()));
            assert(newInsert);

            for (int i = 0; i < node.input_size(); ++i) {
                const std::string input = redirectName(node.input(i));
                (*it).second.push_back(input);
            }

            continue;
            */

            unsigned int nbOutputs = 0;
            for (int i = 0; i < node.input_size(); ++i) {
                const std::string input = redirectName(node.input(i));
                std::shared_ptr<Cell> inputCell = deepNet->getCell(input);
                nbOutputs += inputCell->getNbOutputs();
            }

            std::shared_ptr<Activation> activation
                = Registrar<LinearActivation>::create<Float_T>(model)();

            std::shared_ptr<ActivationCell> activationCell
                = Registrar<ActivationCell>::create<Float_T>(model)(*deepNet,
                                                                    node.output(0),
                                                                    nbOutputs,
                                                                    activation);

            std::vector<std::shared_ptr<Cell> > parentCells;
            for (int i = 0; i < node.input_size(); ++i) {
                const std::string input = redirectName(node.input(i));
                std::shared_ptr<Cell> inputCell = deepNet->getCell(input);
                parentCells.push_back(inputCell);

                activationCell->addInput(inputCell.get());
            }

            deepNet->addCell(activationCell, parentCells);
            activationCell->initialize();
            cell = activationCell;
            continue;
        }
        //ConcatFromSequence
        else if (node.op_type() == "Constant") {
            if ((itAttr = attribute.find("value")) != attribute.end()) {
                initializer[node.output(0)] = &((*itAttr).second->t());
            }
            else if ((itAttr = attribute.find("sparse_value"))
                != attribute.end())
            {
                throw std::runtime_error("Unsupported operation: "
                    "Constant with sparse_value");
            }

            continue;
        }
        //ConstantOfShape
        else if (node.op_type() == "Conv" || node.op_type() == "ConvInteger") {
            if (node.op_type() == "ConvInteger" && node.input_size() > 2) {
                throw std::runtime_error("Unsupported operation: "
                    "ConvInteger with zero point");
            }

            // kernel_shape
            std::vector<unsigned int> kernelDims;

            const std::string inputW = node.input(1);
            unsigned int nbInputs = initializer[inputW]->dims(1);
            const unsigned int nbOutputs = initializer[inputW]->dims(0);

            if ((itAttr = attribute.find("kernel_shape")) != attribute.end()) {
                for (int dim = 0; dim < (*itAttr).second->ints_size(); ++dim)
                    kernelDims.push_back((*itAttr).second->ints(dim));
            }
            else {
                // Deduce kernel size from input W
                for (int dim = 2; dim < initializer[inputW]->dims_size();
                    ++dim)
                {
                    kernelDims.push_back(initializer[inputW]->dims(dim));
                }
            }

            std::reverse(kernelDims.begin(), kernelDims.end());

            // strides
            std::vector<unsigned int> strideDims;

            if ((itAttr = attribute.find("strides")) != attribute.end()) {
                for (int dim = 0; dim < (*itAttr).second->ints_size(); ++dim)
                    strideDims.push_back((*itAttr).second->ints(dim));

                std::reverse(strideDims.begin(), strideDims.end());
            }
            else
                strideDims.resize(kernelDims.size(), 1);

            // dilations
            std::vector<unsigned int> dilationDims;

            if ((itAttr = attribute.find("dilations")) != attribute.end()) {
                for (int dim = 0; dim < (*itAttr).second->ints_size(); ++dim)
                    dilationDims.push_back((*itAttr).second->ints(dim));

                std::reverse(dilationDims.begin(), dilationDims.end());
            }
            else
                dilationDims.resize(kernelDims.size(), 1);

            // pads
            std::vector<int> paddingDimsBegin;
            std::vector<int> paddingDimsEnd;

            if ((itAttr = attribute.find("pads")) != attribute.end()) {
                // Mutually exclusive with auto_pad
                assert((*itAttr).second->ints_size() % 2 == 0);
                const int offset = (*itAttr).second->ints_size() / 2;

                for (int dim = 0; dim < offset; ++dim) {
                    paddingDimsBegin.push_back((*itAttr).second->ints(dim));
                    paddingDimsEnd.push_back((*itAttr).second->ints(offset + dim));
                }
            }
            else if ((itAttr = attribute.find("auto_pad")) != attribute.end()) {
                // Mutually exclusive with pads
                for (unsigned int dim = 0; dim < kernelDims.size(); ++dim) {
                    const int kernelExtent
                        = dilationDims[dim] * (kernelDims[dim] - 1) + 1;
                    const int padding = (kernelExtent - strideDims[dim]);
                    const int floorHalfPadding = (padding / 2);

                    if ((*itAttr).second->s() == "SAME_UPPER") {
                        paddingDimsBegin.push_back(floorHalfPadding);
                        paddingDimsEnd.push_back(padding - floorHalfPadding);
                    }
                    else if ((*itAttr).second->s() == "SAME_LOWER") {
                        paddingDimsBegin.push_back(padding - floorHalfPadding);
                        paddingDimsEnd.push_back(floorHalfPadding);
                    }
                    else if ((*itAttr).second->s() == "VALID") {
                        paddingDimsBegin.push_back(0);
                        paddingDimsEnd.push_back(0);
                    }
                }
            }
            else {
                paddingDimsBegin.resize(kernelDims.size(), 0);
                paddingDimsEnd.resize(kernelDims.size(), 0);
            }

            std::reverse(paddingDimsBegin.begin(), paddingDimsBegin.end());
            std::reverse(paddingDimsEnd.begin(), paddingDimsEnd.end());

            // group
            const int group
                = ((itAttr = attribute.find("group")) != attribute.end())
                    ? (*itAttr).second->i() : 1;

            Tensor<bool> map;

            if (group > 1) {
                const int outSize = nbOutputs / group;
                const int inSize = nbInputs;

                map.resize({nbOutputs, nbInputs * group}, false);

                for (int g = 0; g < group; ++g) {
                    for (int in = 0; in < inSize; ++in) {
                        for (int out = 0; out < outSize; ++out)
                            map(out + g * outSize, in + g * inSize) = true;
                    }
                }
            }

            std::vector<unsigned int> subSampleDims(kernelDims.size(), 1);
            std::shared_ptr<Activation> activation
                = Registrar<LinearActivation>::create<Float_T>(model)();

            // Asymmetric padding
            bool paddingCellRequired = false;

            for (unsigned int dim = 0; dim < paddingDimsBegin.size(); ++dim) {
                if (paddingDimsBegin[dim] != paddingDimsEnd[dim]) {
                    paddingCellRequired = true;
                    break;
                }
            }

            std::vector<int> paddingDims = (paddingCellRequired)
                ? std::vector<int>(kernelDims.size(), 0) : paddingDimsBegin;

            std::vector<unsigned int> kernelDimsCtor(kernelDims);

            if (globTranspose) {
                if (kernelDimsCtor.size() < 2) {
                    kernelDimsCtor.resize(2, 1);
                    subSampleDims.resize(2, 1);
                    strideDims.resize(2, 1);
                    paddingDims.resize(2, 0);
                    dilationDims.resize(2, 1);
                }

                std::swap(kernelDimsCtor[0], kernelDimsCtor[1]);
                std::swap(subSampleDims[0], subSampleDims[1]);
                std::swap(strideDims[0], strideDims[1]);
                std::swap(paddingDims[0], paddingDims[1]);
                std::swap(dilationDims[0], dilationDims[1]);
            }

            // Cell construction
            std::shared_ptr<ConvCell> convCell
                = Registrar<ConvCell>::create<Float_T>(model)(deepNet->getNetwork(),
                                                                *deepNet, 
                                                                node.output(0),
                                                                kernelDimsCtor,
                                                                nbOutputs,
                                                                subSampleDims,
                                                                strideDims,
                                                                paddingDims,
                                                                dilationDims,
                                                                activation);

            // Parameters
            convCell->setParameter<bool>("NoBias", (node.input_size() != 3
                                        || node.op_type() == "ConvInteger"));

            if (iniConfig.currentSection(node.output(0), false)) {
                ConvCellGenerator::generateParams(convCell, iniConfig,
                    node.output(0), model, Float32);
            }
            else if (iniConfig.currentSection(onnxName + ":Conv_def", false)) {
                ConvCellGenerator::generateParams(convCell, iniConfig,
                    onnxName + ":Conv_def", model, Float32);
            }

            const std::string inputX = redirectName(node.input(0));
            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            if (paddingCellRequired) {
                std::cout << "  Added padding: " << paddingDimsBegin
                    << " -- " << paddingDimsEnd << std::endl;

                if (globTranspose) {
                    std::swap(paddingDimsBegin[0], paddingDimsBegin[1]);
                    std::swap(paddingDimsEnd[0], paddingDimsEnd[1]);
                }

                std::shared_ptr<PaddingCell> paddingCell = Registrar
                    <PaddingCell>::create(model)(*deepNet,
                                                node.output(0) + "_padding",
                                                nbInputs * group,
                                                paddingDimsBegin[1],
                                                paddingDimsEnd[1],
                                                paddingDimsBegin[0],
                                                paddingDimsEnd[0]);

                if ((itConcat = concat.find(inputX)) != concat.end()) {
                    for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                        const std::string input = (*itConcat).second[i];
                        std::shared_ptr<Cell> inputCell = getCell(input);
                        parentCells.push_back(inputCell);

                        paddingCell->addInput(inputCell.get());
                    }
                }
                else {
                    std::shared_ptr<Cell> inputXCell = getCell(inputX);
                    parentCells.push_back(inputXCell);

                    if (inputXCell)
                        paddingCell->addInput(inputXCell.get());
                    else {
                        paddingCell->addInput(*sp, 0, 0,
                                              sp->getSizeX(), sp->getSizeY());
                    }
                }

                convCell->addInput(paddingCell.get(), map);

                deepNet->addCell(paddingCell, parentCells);
                paddingCell->initialize();

                parentCells.clear();
                parentCells.push_back(paddingCell);
            }
            else {
                if ((itConcat = concat.find(inputX)) != concat.end()) {
                    for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                        const std::string input = (*itConcat).second[i];
                        std::shared_ptr<Cell> inputCell = getCell(input);
                        parentCells.push_back(inputCell);

                        convCell->addInput(inputCell.get(), map);
                    }
                }
                else {
                    std::shared_ptr<Cell> inputXCell = getCell(inputX);
                    parentCells.push_back(inputXCell);

                    if (inputXCell)
                        convCell->addInput(inputXCell.get(), map);
                    else {
                        convCell->addInput(*sp, 0, 0,
                                           sp->getSizeX(), sp->getSizeY(), map);
                    }
                }
            }

            deepNet->addCell(convCell, parentCells);
            convCell->initialize();
            const std::map<unsigned int, unsigned int> outputsMap 
                                                = convCell->outputsRemap();
            cell = convCell;

            std::cout << "  # Shared synapses: "
                << convCell->getNbSharedSynapses() << std::endl;
            std::cout << "  # Virtual synapses: "
                << convCell->getNbVirtualSynapses() << std::endl;

            //convCell->writeMap("map/" + node.output(0) + "_map.dat");

            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(convCell);

            if(initializeFromONNX) {
                if (cellFrame)
                    cellFrame->keepInSync(false);

                // Free parameters
                if (node.input_size() > 1
                    && (itInit = initializer.find(node.input(1)))
                        != initializer.end())
                {
                    kernelDims.push_back(nbInputs);
                    kernelDims.push_back(convCell->getNbOutputs());

                    Tensor<Float_T> kernels;
                    if ((*itInit).second->data_type() == onnx::TensorProto_DataType_FLOAT) {
                        Tensor<Float_T> wTmp 
                            = ONNX_unpackTensor<Float_T>((*itInit).second, kernelDims);
                        kernels.resize(wTmp.dims());
                        kernels 
                            = ONNX_unpackTensor<Float_T>((*itInit).second, kernelDims);
                    }
                    else if ((*itInit).second->data_type() == onnx::TensorProto_DataType_INT8) {
                        std::cout << Utils::cnotice 
                            << "  Implicit Cast on Weights from INT8 to Float32 for " 
                            << node.op_type()
                            << " for layer type "
                            << Utils::cdef << std::endl;
                        Tensor<int8_t> wTmp 
                            = ONNX_unpackTensor<int8_t>((*itInit).second, kernelDims);
                        kernels.resize(wTmp.dims());
                        for(unsigned int i = 0; i < wTmp.size(); ++i) {
                            kernels(i) = (float) wTmp(i);
                        }
                    }
                    else if ((*itInit).second->data_type() == onnx::TensorProto_DataType_DOUBLE) {
                        std::cout << Utils::cnotice 
                            << "  Implicit Cast on Weights from Float64 to Float32 for " 
                            << node.op_type()
                            << " for layer type "
                            << Utils::cdef << std::endl;
                        Tensor<double> wTmp 
                            = ONNX_unpackTensor<double>((*itInit).second, kernelDims);
                        kernels.resize(wTmp.dims());
                        for(unsigned int i = 0; i < wTmp.size(); ++i) {
                            kernels(i) = (float) wTmp(i);
                        }
                    }
                    else {
                        throw std::runtime_error("Unsupported datatype: "
                            "Conv or ConvInteger Layer only support Float32, Float64 or INT8 Weights");
                    }

                    for (unsigned int output = 0;
                        output < convCell->getNbOutputs(); ++output)
                    {
                        const unsigned int outputRemap = (!outputsMap.empty())
                            ? outputsMap.find(output)->second : output;
                        for (unsigned int channel = 0;
                            channel < convCell->getNbChannels(); ++channel)
                        {
                            if (!convCell->isConnection(channel, outputRemap))
                                continue;

                            if (globTranspose) {
                                Tensor<Float_T> kernel
                                    = kernels[output][channel / group];

                                if (kernel.nbDims() < 2)
                                    kernel.reshape({kernel.dims()[0], 1});

                                Tensor<Float_T> transKernel(
                                    {kernel.dimY(), kernel.dimX()});

                                for (unsigned int sx = 0; sx < kernel.dimX();
                                    ++sx)
                                {
                                    for (unsigned int sy = 0; sy < kernel.dimY();
                                        ++sy)
                                    {
                                        transKernel(sy, sx) = kernel(sx, sy);
                                    }
                                }

                                convCell->setWeight(outputRemap, channel, transKernel);
                            }
                            else {
                                convCell->setWeight(outputRemap, channel,
                                            kernels[output][channel / group]);
                            }
                        }
                    }
                }
                else if (node.input_size() > 1) {
                    std::cout << "  No initializer for \"" << node.input(1)
                        << "\"" << std::endl;
                }

                if (!convCell->getParameter<bool>("NoBias")) {
                    if (node.input_size() > 2
                        && (itInit = initializer.find(node.input(2)))
                            != initializer.end())
                    {
                        Tensor<Float_T> biases;
                        if ((*itInit).second->data_type() == onnx::TensorProto_DataType_FLOAT) {
                            biases.resize({(unsigned int)convCell->getNbOutputs()});
                            biases 
                                = ONNX_unpackTensor<Float_T>((*itInit).second,
                                                            {(unsigned int)convCell->getNbOutputs()});
                        }
                        else if ((*itInit).second->data_type() == onnx::TensorProto_DataType_INT32) {
                            std::cout << Utils::cnotice 
                                << "  Implicit Cast on Biases from INT32 to Float32 for " 
                                << node.op_type()
                                << " for layer type "
                                << Utils::cdef << std::endl;
                            Tensor<int32_t> bTmp 
                                = ONNX_unpackTensor<int32_t>((*itInit).second,
                                                            {(unsigned int)convCell->getNbOutputs()});
                            biases.resize(bTmp.dims());
                            for(unsigned int i = 0; i < bTmp.size(); ++i) {
                                biases(i) = (float) bTmp(i);
                            }
                        }
                        else if ((*itInit).second->data_type() == onnx::TensorProto_DataType_DOUBLE) {
                            std::cout << Utils::cnotice 
                                << "  Implicit Cast on Biases from Float64 to Float32 for " 
                                << node.op_type()
                                << " for layer type "
                                << Utils::cdef << std::endl;
                            Tensor<double> bTmp 
                                = ONNX_unpackTensor<double>((*itInit).second, kernelDims);
                            biases.resize(bTmp.dims());
                            for(unsigned int i = 0; i < bTmp.size(); ++i) {
                                biases(i) = (float) bTmp(i);
                            }
                        }
                        else {
                            throw std::runtime_error("Unsupported datatype: "
                                "Conv or ConvInteger Layer only support Float32, Float64 or INT32 Biases");
                        }
                        biases.reshape({1, convCell->getNbOutputs()}); // Adding an empty dim to avoid error when accessing bias
                        for (unsigned int output = 0;
                            output < convCell->getNbOutputs(); ++output)
                        {
                            const unsigned int outputRemap = (!outputsMap.empty())
                                ? outputsMap.find(output)->second : output;
                            convCell->setBias(outputRemap, biases[output]);
                        }
                    }
                    else if (node.input_size() > 2) {
                        std::cout << "  No initializer for \"" << node.input(2)
                            << "\"" << std::endl;
                    }
                }
                else if (node.input_size() > 2) {
                    std::cout << Utils::cwarning << "  Biases in ONNX ignored!"
                        << Utils::cdef << std::endl;
                }
                if (cellFrame)
                    cellFrame->synchronizeToD(true);
            }
        }
        //else if (node.op_type() == "ConvTranspose") {

        //}
        //Cos
        //Cosh
        //CumSum
        //DepthToSpace
        //DequantizeLinear
        //Det
        //Div -> see Sum
        else if (node.op_type() == "Dropout") {
            float ratio = 0.5;

            if ((itAttr = attribute.find("ratio")) != attribute.end())
                ratio = (*itAttr).second->f();

            const std::string inputX = redirectName(node.input(0));
            unsigned int nbOutputs = 0;
            std::vector<size_t> inputsDims;

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                    const std::string input = (*itConcat).second[i];
                    std::shared_ptr<Cell> inputCell = getCell(input);

                    nbOutputs += inputCell->getNbOutputs();
                    inputsDims = inputCell->getOutputsDims();
                }
            }
            else {
                std::shared_ptr<Cell> inputXCell = getCell(inputX);
                
                if (inputXCell) {
                    nbOutputs += inputXCell->getNbOutputs();
                    inputsDims = inputXCell->getOutputsDims();
                }
            }

            std::vector<std::shared_ptr<Cell> > parentCells;

            std::shared_ptr<DropoutCell> dropoutCell
                = Registrar<DropoutCell>::create<Float_T>(model)(*deepNet, 
                                                                node.output(0),
                                                                nbOutputs);

            dropoutCell->setParameter<double>("Dropout", ratio);

            if (iniConfig.currentSection(node.output(0), false)) {
                DropoutCellGenerator::generateParams(dropoutCell, iniConfig,
                    node.output(0), model, Float32);
            }
            else if (iniConfig.currentSection(onnxName + ":Dropout_def", false)) {
                DropoutCellGenerator::generateParams(dropoutCell, iniConfig,
                    onnxName + ":Dropout_def", model, Float32);
            }

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                    const std::string input = (*itConcat).second[i];
                    std::shared_ptr<Cell> inputCell = getCell(input);
                    parentCells.push_back(inputCell);

                    dropoutCell->addInput(inputCell.get());
                }
            }
            else {
                std::shared_ptr<Cell> inputXCell = getCell(inputX);
                parentCells.push_back(inputXCell);

                if (inputXCell)
                    dropoutCell->addInput(inputXCell.get());
                else {
                    dropoutCell->addInput(*sp, 0, 0,
                                        sp->getSizeX(), sp->getSizeY());
                }
            }

            deepNet->addCell(dropoutCell, parentCells);
            dropoutCell->initialize();
            cell = dropoutCell;
        }
        //Elu
        //Equal
        //Erf
        //Exp
        //Expand
        //EyeLike
        else if (node.op_type() == "Flatten") {
            std::cout << Utils::cnotice << "  Ignore Flatten operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //Floor
        //GRU
        //Gather
        else if (node.op_type() == "Gather") {
            std::cout << Utils::cnotice << "  Ignore Gather operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //GatherElements
        //GatherND
        else if (node.op_type() == "Gemm" || node.op_type() == "MatMul"
            || node.op_type() == "MatMulInteger")
        {
            if (node.op_type() == "MatMulInteger" && node.input_size() > 2) {
                throw std::runtime_error("Unsupported operation: "
                    "MatMulInteger with zero point");
            }

            const std::string inputData1 = redirectName(node.input(0));
            const std::string inputData2 = redirectName(node.input(1));
            std::string inputData;

            if ((itInit = initializer.find(inputData1)) != initializer.end()
                && initializer.find(inputData2) == initializer.end())
            {
                inputData = inputData2;
            }
            else if ((itInit = initializer.find(inputData2))
                    != initializer.end()
                && initializer.find(inputData1) == initializer.end())
            {
                inputData = inputData1;
            }

            // Attribute are for Gemm only
            float alpha = 1.0;
            float beta = 1.0;
            //bool transA = false;
            bool transB = false;

            if (node.op_type() == "Gemm" || node.op_type() == "MatMulInteger") {
                if ((itAttr = attribute.find("alpha")) != attribute.end())
                    alpha = (*itAttr).second->f();

                if ((itAttr = attribute.find("beta")) != attribute.end())
                    beta = (*itAttr).second->f();

                //if ((itAttr = attribute.find("transA")) != attribute.end())
                //    transA = (*itAttr).second->i();

                if ((itAttr = attribute.find("transB")) != attribute.end())
                    transB = (*itAttr).second->i();
            }

            if (!inputData.empty()) {
                Tensor<Float_T> weights;
                if ((*itInit).second->data_type() == onnx::TensorProto_DataType_FLOAT) {
                    Tensor<Float_T> wTmp = ONNX_unpackTensor<Float_T>((*itInit).second);
                    weights.resize(wTmp.dims());
                    weights 
                        = ONNX_unpackTensor<Float_T>((*itInit).second);
                }
                else if ((*itInit).second->data_type() == onnx::TensorProto_DataType_INT8) {
                    std::cout << Utils::cnotice 
                        << "  Implicit Cast on Weights from INT8 to Float32 for " 
                        << node.op_type()
                        << " for layer type "
                        << Utils::cdef << std::endl;
                    Tensor<int8_t> wTmp = ONNX_unpackTensor<int8_t>((*itInit).second);
                    weights.resize(wTmp.dims());
                    for(unsigned int i = 0; i < wTmp.size(); ++i) {
                        weights(i) = (float) wTmp(i);
                    }
                }
                else if ((*itInit).second->data_type() == onnx::TensorProto_DataType_DOUBLE) {
                    std::cout << Utils::cnotice 
                        << "  Implicit Cast on Weights from Float64 to Float32 for " 
                        << node.op_type()
                        << " for layer type "
                        << Utils::cdef << std::endl;
                    Tensor<double> wTmp = ONNX_unpackTensor<double>((*itInit).second);
                    weights.resize(wTmp.dims());
                    for(unsigned int i = 0; i < wTmp.size(); ++i) {
                        weights(i) = (float) wTmp(i);
                    }
                }
                else {
                    throw std::runtime_error("Unsupported datatype: "
                        "Gemm or MatMul or MatMulInteger Layer only support Float32, Float64 or INT8 Weights");
                }

                if ((itShape = shape.find((*itInit).first)) != shape.end())
                    weights.reshape((*itShape).second);

                const unsigned int nbOutputs = (transB)
                    ? weights.dimB() : weights.size() / weights.dimB();

                std::map<std::string, std::vector<std::string> >
                    ::const_iterator itConcat;
                std::vector<std::shared_ptr<Cell> > parentCells;

                std::shared_ptr<Activation> activation
                    = Registrar<LinearActivation>::create<Float_T>(model)();

                std::shared_ptr<FcCell> fcCell
                    = Registrar<FcCell>::create<Float_T>(model)(deepNet->getNetwork(),
                                                                *deepNet, 
                                                                node.output(0),
                                                                nbOutputs,
                                                                activation);

                if (!(node.op_type() == "Gemm" && node.input_size() > 2))
                    fcCell->setParameter<bool>("NoBias", true);

                if (iniConfig.currentSection(node.output(0), false)) {
                    FcCellGenerator::generateParams(fcCell, iniConfig,
                        node.output(0), model, Float32);
                }
                else if (iniConfig.currentSection(onnxName + ":Fc_def", false)) {
                    FcCellGenerator::generateParams(fcCell, iniConfig,
                        onnxName + ":Fc_def", model, Float32);
                }

                if ((itConcat = concat.find(inputData)) != concat.end()) {
                    for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                        const std::string input = (*itConcat).second[i];
                        std::shared_ptr<Cell> inputCell = getCell(input);
                        parentCells.push_back(inputCell);

                        fcCell->addInput(inputCell.get());
                    }
                }
                else {
                    std::shared_ptr<Cell> inputDataCell = getCell(inputData);
                    parentCells.push_back(inputDataCell);

                    if (inputDataCell)
                        fcCell->addInput(inputDataCell.get());
                    else {
                        fcCell->addInput(*sp, 0, 0,
                                            sp->getSizeX(), sp->getSizeY());
                    }
                }

                deepNet->addCell(fcCell, parentCells);
                fcCell->initialize();
                cell = fcCell;

                std::cout << "  # Synapses: " << fcCell->getNbSynapses()
                    << std::endl;
    
                if (fcCell->getInputsSize() != weights.size() / nbOutputs)
                {
                    std::ostringstream errorStr;
                    errorStr << "Unsupported operation: "
                        << node.op_type() << " with weights size mismatch."
                        " Inputs dims: " << fcCell->getInputsDims()
                        << ", weights dims: " << weights.dims()
                        << ", nb. outputs: " << nbOutputs;

                    throw std::runtime_error(errorStr.str());
                }
                if(initializeFromONNX) {

                    // Init weights
                    if (transB) {
                        weights.reshape({1, fcCell->getInputsSize(),
                                        fcCell->getNbOutputs()});
                    }
                    else {
                        weights.reshape({1, fcCell->getNbOutputs(),
                                        fcCell->getInputsSize()});
                    }

                    std::shared_ptr<Cell_Frame_Top> cellFrame
                        = std::dynamic_pointer_cast<Cell_Frame_Top>(fcCell);

                    if (cellFrame)
                        cellFrame->keepInSync(false);

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (weights.size() > 1024)
#else
#pragma omp parallel for if (weights.size() > 1024)
#endif
                    for (int output = 0;
                        output < (int)fcCell->getNbOutputs(); ++output)
                    {
                        for (unsigned int ch = 0; ch < fcCell->getNbChannels(); ++ch) {
                            for (unsigned int iy = 0; iy < fcCell->getChannelsHeight(); ++iy) {
                                for (unsigned int ix = 0; ix < fcCell->getChannelsWidth(); ++ix) {
                                    const unsigned int channel = (globTranspose)
                                        ? iy + fcCell->getChannelsHeight() * (ix + fcCell->getChannelsWidth() * ch)
                                        : ix + fcCell->getChannelsWidth() * (iy + fcCell->getChannelsHeight() * ch);

                                    Tensor<Float_T> w = (transB)
                                        ? weights[output][channel]
                                        : weights[channel][output];

                                    if (alpha != 1.0) {
                                        for (unsigned int i = 0; i < w.size(); ++i)
                                            w(i) *= alpha;
                                    }

                                    fcCell->setWeight(output, channel, w);
                                }
                            }
                        }
                    }

                    // Init bias (Gemm only)
                    if (node.op_type() == "Gemm" && node.input_size() > 2) {
                        if (!fcCell->getParameter<bool>("NoBias")) {
                            if ((itInit = initializer.find(node.input(2)))
                                != initializer.end())
                            {
                                Tensor<Float_T> bias
                                    = ONNX_unpackTensor<Float_T>((*itInit).second,
                                        {(unsigned int)fcCell->getNbOutputs()});
                                bias.reshape({1, fcCell->getNbOutputs()});

                                for (unsigned int output = 0;
                                    output < fcCell->getNbOutputs(); ++output)
                                {
                                    if (beta != 1.0)
                                        bias[output](0) *= beta;

                                    fcCell->setBias(output, bias[output]);
                                }
                            }
                            else {
                                std::cout << "  No initializer for \""
                                    << node.input(2) << "\"" << std::endl;
                            }
                        }
                        else {
                            std::cout << Utils::cwarning
                                << "  Biases in ONNX ignored!"
                                << Utils::cdef << std::endl;
                        }
                    }

                    if (cellFrame)
                        cellFrame->synchronizeToD(true);
                }
            }
            else {
                throw std::runtime_error("Unsupported operation: "
                    + node.op_type() + " without Cell");
            }
        }
        //GlobalAveragePool -> see AveragePool
        //GlobalLpPool
        //GlobalMaxPool -> see AveragePool
        //Greater
        //HardSigmoid
        //Hardmax
        else if (node.op_type() == "Identity") {
            std::cout << Utils::cnotice << "  Ignore Identity operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //If
        //InstanceNormalization
        //IsInf
        //IsNaN
        else if (node.op_type() == "LRN") {
            float alpha = 0.0001;
            float beta = 0.75;
            float bias = 1.0;
            int size = 0;

            if ((itAttr = attribute.find("alpha")) != attribute.end())
                alpha = (*itAttr).second->f();

            if ((itAttr = attribute.find("beta")) != attribute.end())
                beta = (*itAttr).second->f();

            if ((itAttr = attribute.find("bias")) != attribute.end())
                bias = (*itAttr).second->f();

            if ((itAttr = attribute.find("size")) != attribute.end())
                size = (*itAttr).second->i();
            else {
                throw std::runtime_error("Missing required attribute:"
                                         " size for LRN");
            }

            const std::string inputX = redirectName(node.input(0));
            std::shared_ptr<Cell> inputXCell = getCell(inputX);

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            std::shared_ptr<LRNCell> lrnCell
                = Registrar<LRNCell>::create<Float_T>(model)(*deepNet, 
                                                                node.output(0),
                                                                inputXCell->getNbOutputs());

            lrnCell->setParameter<double>("Alpha", alpha);
            lrnCell->setParameter<double>("Beta", beta);
            lrnCell->setParameter<double>("K", bias);
            lrnCell->setParameter<unsigned int>("N", size);

            if (iniConfig.currentSection(node.output(0), false)) {
                LRNCellGenerator::generateParams(lrnCell, iniConfig,
                    node.output(0), model, Float32);
            }
            else if (iniConfig.currentSection(onnxName + ":LRN_def", false)) {
                LRNCellGenerator::generateParams(lrnCell, iniConfig,
                    onnxName + ":LRN_def", model, Float32);
            }

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                throw std::runtime_error("Unsupported operation: Concat before "
                    "LRN");
            }
            else {
                std::shared_ptr<Cell> inputXCell = getCell(inputX);
                parentCells.push_back(inputXCell);

                if (inputXCell)
                    lrnCell->addInput(inputXCell.get());
                else {
                    lrnCell->addInput(*sp, 0, 0,
                                        sp->getSizeX(), sp->getSizeY());
                }
            }

            deepNet->addCell(lrnCell, parentCells);
            lrnCell->initialize();
            cell = lrnCell;
        }
        //LSTM
        else if (node.op_type() == "LeakyRelu") {
            float alpha = 0.01;  // default ONNX value

            if ((itAttr = attribute.find("alpha")) != attribute.end())
                alpha = (*itAttr).second->f();

            std::shared_ptr<Activation> activation
                = Registrar<RectifierActivation>::create<Float_T>(model)();
            activation->setParameter<double>("LeakSlope", alpha);

            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (cellFrame->getActivation()
                && cellFrame->getActivation()->getType()
                    != LinearActivation::Type)
            {
                throw std::runtime_error("Cell " + cell->getName()
                    + " already has an activation!");
            }

            cellFrame->setActivation(activation);

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //Less
        //Log
        //LogSoftmax
        //Loop
        //LpNormalization
        //LpPool
        //MatMul -> see Gemm
        //Max -> see Sum
        //MaxPool -> see AveragePool
        //MaxRoiPool
        //MaxUnpool
        //Mean
        //Min
        //Mod
        //Mul -> see Sum
        //Multinomial
        //Neg
        //NonMaxSuppression
        //NonZero
        //Not
        //OneHot
        //Or
        //PRelu
        else if (node.op_type() == "Pad") {
            std::string mode = "constant";

            if ((itAttr = attribute.find("mode")) != attribute.end())
                mode = (*itAttr).second->s();

            if (mode != "constant") {
                throw std::runtime_error("Only \"constant\" mode is supported"
                    " for Pad operator");
            }

            if (node.input_size() > 2) {
                throw std::runtime_error("\"constant_value\" input is not"
                    " supported for Pad operator.");
            }

            std::vector<int> paddingDimsBegin;
            std::vector<int> paddingDimsEnd;
            if (node.input_size() > 1) {
                if ((itInit = initializer.find(node.input(1))) != initializer.end())
                {
                    Tensor<int64_t> pad
                        = ONNX_unpackTensor<int64_t>((*itInit).second);

                assert(pad.size() % 2 == 0);
                const int offset = pad.size() / 2;

                for (int dim = 0; dim < offset; ++dim) {
                    paddingDimsBegin.push_back(pad(dim));
                    paddingDimsEnd.push_back(pad(offset + dim));
                }
            }
            else {
                std::stringstream msgStr;
                msgStr << "  No initializer for \"" << node.input(1)
                    << "\"" << std::endl;

                throw std::runtime_error(msgStr.str());
            }

            //assert(pad.size() % 2 == 0);
            //const int offset = pad.size() / 2;

            //for (int dim = 0; dim < offset; ++dim) {
            //    paddingDimsBegin.push_back(pad(dim));
            //    paddingDimsEnd.push_back(pad(offset + dim));
           // }
            std::reverse(paddingDimsBegin.begin(), paddingDimsBegin.end());
            std::reverse(paddingDimsEnd.begin(), paddingDimsEnd.end());

            const std::string inputX = redirectName(node.input(0));
            std::shared_ptr<Cell> inputXCell = getCell(inputX);

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            if (globTranspose) {
                std::swap(paddingDimsBegin[0], paddingDimsBegin[1]);
                std::swap(paddingDimsEnd[0], paddingDimsEnd[1]);
            }

            std::shared_ptr<PaddingCell> paddingCell = Registrar
                <PaddingCell>::create(model)(*deepNet,
                                            node.output(0),
                                            inputXCell->getNbOutputs(),
                                            paddingDimsBegin[1],
                                            paddingDimsEnd[1],
                                            paddingDimsBegin[0],
                                            paddingDimsEnd[0]);

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                    const std::string input = (*itConcat).second[i];
                    std::shared_ptr<Cell> inputCell = getCell(input);
                    parentCells.push_back(inputCell);

                    paddingCell->addInput(inputCell.get());
                }
            }
            else {
                std::shared_ptr<Cell> inputXCell = getCell(inputX);
                parentCells.push_back(inputXCell);

                if (inputXCell)
                    paddingCell->addInput(inputXCell.get());
                else {
                    paddingCell->addInput(*sp, 0, 0,
                                        sp->getSizeX(), sp->getSizeY());
                }
            }

            deepNet->addCell(paddingCell, parentCells);
            paddingCell->initialize();
            cell = paddingCell;
            continue;
               // }
           }
            std::cout << "  No initializer for Padding operation, it will be ignored" << std::endl;

            std::cout << Utils::cnotice << "  Ignore Padding operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //Pow
        //QLinearConv
        //QLinearMatMul
        //QuantizeLinear
        //RNN
        //RandomNormal
        //RandomNormalLike
        //RandomUniform
        //RandomUniformLike
        //Reciprocal
        //ReduceL1
        //ReduceL2
        //ReduceLogSum
        //ReduceLogSumExp
        //ReduceMax
        //ReduceMean
        //ReduceMin
        //ReduceProd
        //ReduceSum
        //ReduceSumSquare
        else if (node.op_type() == "Relu") {

            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (cellFrame->getActivation()
                && cellFrame->getActivation()->getType()
                    != LinearActivation::Type)
            {
                if (cellFrame->getActivation()->getType()
                    == RectifierActivation::Type)
                {
                    // If there is already a ReLU, don't change it as it may 
                    // include clipping
                    std::cout << Utils::cnotice << "  Ignore Relu operation as"
                        " there is already a Relu/clipping" << Utils::cdef 
                        << std::endl;
                }
                else {
                    throw std::runtime_error("Cell " + cell->getName()
                        + " already has an activation!");
                }
            }
            else {

                if (iniConfig.currentSection(node.output(0), false)) {
                    ActivationGenerator::generateParams(cellFrame, iniConfig,
                        node.output(0), model, Float32);
                }
                else {
                    cellFrame->setActivation(Registrar<RectifierActivation>
                        ::create<Float_T>(model)());
                }
                /*
                else
                if (iniConfig.currentSection(onnxName + ":Rectifier_def", false)) {
                    ActivationGenerator::generateParams(cellFrame, iniConfig,
                        onnxName + ":Rectifier_def", model, Float32);
                }*/

            }

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;


/*
            const std::string inputX = redirectName(node.input(0));
            std::shared_ptr<Cell> inputXCell
                = (deepNet->getCells().empty())
                    ? std::shared_ptr<Cell>()
                    : deepNet->getCell(inputX);

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;


            std::shared_ptr<Activation> activation
                = Registrar<RectifierActivation>::create<Float_T>(model)();

            std::shared_ptr<ActivationCell> activationCell
                = Registrar<ActivationCell>::create<Float_T>(model)(*deepNet,
                                                                    node.output(0),
                                                                    (unsigned int) inputXCell->getNbOutputs(),
                                                                    activation);
            
            if (iniConfig.currentSection(node.output(0), false)) {
                ActivationCellGenerator::generateParams(activationCell, iniConfig,
                    node.output(0), model, Float32);
            }
            else
            if (iniConfig.currentSection(onnxName + ":Rectifier_def", false)) {
                ActivationCellGenerator::generateParams(activationCell, iniConfig,
                    onnxName + ":Rectifier_def", model, Float32);
            }


            if ((itConcat = concat.find(inputX)) != concat.end()) {
                for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                    const std::string input = (*itConcat).second[i];
                    std::shared_ptr<Cell> inputCell = deepNet->getCell(input);
                    parentCells.push_back(inputCell);

                    activationCell->addInput(inputCell.get());
                }
            }
            else {
                    
                std::shared_ptr<Cell> inputXCell
                    = (deepNet->getCells().empty())
                        ? std::shared_ptr<Cell>()
                        : deepNet->getCell(inputX);
                parentCells.push_back(inputXCell);

                if (inputXCell)
                    activationCell->addInput(inputXCell.get());
                else {
                    activationCell->addInput(*sp, 0, 0,
                                        sp->getSizeX(), sp->getSizeY());
                }
            }
            deepNet->addCell(activationCell, parentCells);
            activationCell->initialize();
            cell = activationCell;
            continue;
*/

        }
        else if (node.op_type() == "Reshape") {
            const std::string inputX = redirectName(node.input(0));

            std::vector<size_t> newShape;

            if (node.input_size() > 1) {
                // see https://github.com/onnx/onnx/pull/608
                if ((itInit = initializer.find(node.input(1)))
                    != initializer.end())
                {
                    const Tensor<int64_t> shapeTensor
                        = ONNX_unpackTensor<int64_t>((*itInit).second);

                    for (int dim = 0; dim < (int)shapeTensor.size(); ++dim)
                        newShape.push_back(shapeTensor(dim));
                }
                // if no initializer is found, the shape is non-constant and
                // computed by previous layers.
            }
            else if ((itAttr = attribute.find("shape")) != attribute.end()) {
                for (int dim = 0; dim < (*itAttr).second->ints_size(); ++dim)
                    newShape.push_back((*itAttr).second->ints(dim));

                //Tensor<int64_t> shapeTensor
                //    = ONNX_unpackTensor<int64_t>(&((*itAttr).second->t()));
                //newShape = shapeTensor.data();
            }

            if (newShape.empty()) {
                std::cout << Utils::cnotice << "  Ignore Reshape operation"
                    << " with non-constant shape" << Utils::cdef << std::endl;

                std::cout << "  " << node.output(0) << " -> "
                    << inputX << std::endl;
                redirect[node.output(0)] = inputX;
                continue;
            }

            std::reverse(newShape.begin(), newShape.end());

            if ((itInit = initializer.find(inputX)) != initializer.end()) {
                shape[inputX] = newShape;

                std::cout << "  " << node.output(0) << " -> "
                    << inputX << std::endl;
                redirect[node.output(0)] = inputX;
                continue;
            }
            else {
                const unsigned int nbOutputs = newShape[2];

                std::map<std::string, std::vector<std::string> >
                    ::const_iterator itConcat;
                std::vector<std::shared_ptr<Cell> > parentCells;

                if (globTranspose)
                    std::swap(newShape[0], newShape[1]);

                std::shared_ptr<ReshapeCell> reshapeCell
                    = Registrar<ReshapeCell>::create<Float_T>(model)(*deepNet, 
                        node.output(0),
                        nbOutputs,
                        std::vector<int>(newShape.begin(), newShape.end()));

                if ((itConcat = concat.find(inputX)) != concat.end()) {
                    throw std::runtime_error("Unsupported operation: Concat before "
                        "Reshape");
                }
                else {
                    std::shared_ptr<Cell> inputXCell = getCell(inputX);
                    parentCells.push_back(inputXCell);

                    if (inputXCell)
                        reshapeCell->addInput(inputXCell.get());
                    else {
                        reshapeCell->addInput(*sp, 0, 0,
                                            sp->getSizeX(), sp->getSizeY());
                    }
                }

                deepNet->addCell(reshapeCell, parentCells);
                reshapeCell->initialize();
                cell = reshapeCell;
            }
        }
        else if (node.op_type() == "Resize") {
            //Default mode set to NearestNeighbor
            ResizeCell::ResizeMode resizeMode 
                = ResizeCell::ResizeMode::NearestNeighbor;
            //Default mode is aligned corner set to false
            bool alignCorners = false;
            std::vector<size_t>  inputsDims;       

            if ((itAttr = attribute.find("coordinate_transformation_mode")) 
                    != attribute.end()) {

                if((*itAttr).second->s() == "asymmetric") {
                    alignCorners = false;
                    std::cout << Utils::cnotice << ""
                        << "   Resize Mode for Coordinate: [Asymmetric]" 
                        << Utils::cdef << std::endl;

                }
                else if((*itAttr).second->s() == "align_corners") {
                    alignCorners = true;
                    std::cout << Utils::cnotice 
                        << "   Resize Mode for Coordinate: [Aligned Corner]" 
                        << Utils::cdef << std::endl;
                }
                else {
                        std::cout << Utils::cnotice  
                                << (*itAttr).second->s()
                                << "   Resize Mode for Coordinate: [" << (*itAttr).second->s() 
                                << "] not yet supported by N2D2, back to default mode"
                            << Utils::cdef << std::endl;
                        std::cout << Utils::cnotice << ""
                            << "   Resize Mode for Coordinate: [Asymmetric]" 
                            << Utils::cdef << std::endl;
                }
            }

            if ((itAttr = attribute.find("cubic_coeff_a")) != attribute.end()) {
                std::cout << Utils::cnotice  
                        << "   Resize Parameter: [cubic_coeff_a]"  
                        << " not yet supported by N2D2"
                    << Utils::cdef << std::endl;
            }

            if ((itAttr = attribute.find("mode")) != attribute.end()) {
                if((*itAttr).second->s() == "nearest") {
                    std::cout << Utils::cnotice << ""
                        << "   Resize Mode for Interpolation: [NearestNeighbor]" 
                        << Utils::cdef << std::endl;
                }
                else if((*itAttr).second->s()  == "linear") {
                    resizeMode 
                        = ResizeCell::ResizeMode::BilinearTF;
                    std::cout << Utils::cnotice << ""
                        << "   Resize Mode for Interpolation: [BilinearTF]" 
                        << Utils::cdef << std::endl;
                }
                else {
                    std::cout << Utils::cnotice  
                            << "   Resize Mode for Coordinate: [" << (*itAttr).second->s() 
                            << "] not yet supported by N2D2, back to default mode"
                        << Utils::cdef << std::endl;
                    std::cout << Utils::cnotice  
                            << (*itAttr).second->s()
                            << "  is not yet supported by N2D2, "
                            << " back to default interpolation resize mode => Nearest Neighbor"
                        << Utils::cdef << std::endl;
                }
            }
            std::size_t resizeDimX = 0;
            std::size_t resizeDimY = 0;

            const std::string inputX = redirectName(node.input(0));
            std::shared_ptr<Cell> inputXCell = getCell(inputX);

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;
            if (node.input_size() > 1 && (itInit = initializer.find(redirectName(node.input(1)))) != initializer.end()) {
                const Tensor<float> roiTensor
                            = ONNX_unpackTensor<float>((*itInit).second);
                if(!roiTensor.empty()) {
                    std::cout << "   Resize from [scales] " << 
                                    "===> dimensions X [" << resizeDimX 
                                    << "] and Y [" << resizeDimY << "]" << std::endl; 
                    throw std::runtime_error("Resize from ROI maps is not yet"
                        " supported by N2D2.");
                }
            }
            if (node.input_size() > 2 && (itInit = initializer.find(redirectName(node.input(2)))) != initializer.end()) {
                const Tensor<float> scalesTensor
                            = ONNX_unpackTensor<float>((*itInit).second);
                if(!scalesTensor.empty()) {
                    inputsDims = inputXCell->getOutputsDims();
                    resizeDimX = std::rintf(inputsDims[0]*scalesTensor(3));
                    resizeDimY = std::rintf(inputsDims[1]*scalesTensor(2));
                    std::cout << "   Resize from [scales] " << 
                                    "===> dimensions X [" << resizeDimX 
                                    << "] and Y [" << resizeDimY << "]" << std::endl; 
                }
            }
            if (node.input_size() > 3) {
                const std::string inputSizes 
                    = redirectName(node.input(3));

                //Todo : Improve the minigraph handling for sizes from input
                if ((itConcat = concat.find(inputSizes)) != concat.end()) {
                    for (unsigned int i = 0; i < (*itConcat).second.size(); ++i) {
                        const std::string input = redirectName((*itConcat).second[i]);
                        std::map<std::string, std::vector<std::string> >
                            ::const_iterator itConcat2ndDim;
                        if ((itConcat2ndDim = concat.find(input)) != concat.end()) {
                            for (unsigned int i = 0; i < (*itConcat2ndDim).second.size(); ++i) {
                                const std::string input2nd = redirectName((*itConcat2ndDim).second[i]);
                                std::map<std::string, std::vector<std::string> >
                                    ::const_iterator itConcat3rddDim;
                                if ((itConcat3rddDim = concat.find(input2nd)) != concat.end()) {
                                    for (unsigned int i = 0; i < (*itConcat3rddDim).second.size(); ++i) {
                                        const std::string input3rd 
                                            = redirectName((*itConcat3rddDim).second[i]);
                                        std::shared_ptr<Cell> inputCell3rd = getCell(input3rd);
                                        inputsDims = inputCell3rd->getOutputsDims();
                                    }
                                }
                            }
                        }
                    }
                    resizeDimX = inputsDims[0];
                    resizeDimY = inputsDims[1];
                    std::cout << "   Resize from [minigraph] " << 
                        "===> dimensions X [" << resizeDimX 
                                    << "] and Y [" << resizeDimY << "]" << std::endl; 
                } 
                else {
                    itInit = initializer.find(redirectName(node.input(3)));
                    const Tensor<int64_t> sizesTensor
                                = ONNX_unpackTensor<int64_t>((*itInit).second);
                    if(!sizesTensor.empty()) {
                        resizeDimX = sizesTensor(3);
                        resizeDimY = sizesTensor(2);
                        std::cout << "   Resize from [sizes] " << 
                                        "===> dimensions X [" << resizeDimX 
                                        << "] and Y [" << resizeDimY << "]" << std::endl; 

                    }
                }
            }

            std::shared_ptr<ResizeCell> resizeCell
                = Registrar<ResizeCell>::create(model)(*deepNet, 
                                                                node.output(0),
                                                                resizeDimX,
                                                                resizeDimY,
                                                                inputXCell->getNbOutputs(),
                                                                resizeMode);

            resizeCell->setParameter<bool>("AlignCorners", alignCorners);

            parentCells.push_back(inputXCell);

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                throw std::runtime_error("Unsupported operation: Concat before "
                    "ResizeCell");
            }
            else {
                if (inputXCell)
                    resizeCell->addInput(inputXCell.get());
                else {
                    resizeCell->addInput(*sp, 0, 0,
                                        sp->getSizeX(), sp->getSizeY());
                }
            }
            deepNet->addCell(resizeCell, parentCells);

            resizeCell->initialize();
            cell = resizeCell;
        }

        //}
        //ReverseSequence
        //RoiAlign
        //Round
        //Scan
        //Scatter
        //ScatterElements
        //ScatterND
        //Selu
        //SequenceAt
        //SequenceConstruct
        //SequenceEmpty
        else if (node.op_type() == "SequenceEmpty") {
            std::cout << Utils::cnotice << "  Ignore SequenceEmpty operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //SequenceErase
        //SequenceInsert
        //SequenceLength
        else if (node.op_type() == "Shape") {
            std::cout << Utils::cnotice << "  Ignore Shape operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //Shrink
        else if (node.op_type() == "Sigmoid") {
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (cellFrame->getActivation()
                && cellFrame->getActivation()->getType()
                    != LinearActivation::Type)
            {
                throw std::runtime_error("Cell " + cell->getName()
                    + " already has an activation!");
            }

            cellFrame->setActivation(Registrar<LogisticActivation>
                ::create<Float_T>(model)(false));

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //Sign
        //Sin
        //Sinh
        //Size
        //Slice
        else if (node.op_type() == "Slice") {
            std::cout << Utils::cnotice << "  Ignore Slice operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        else if (node.op_type() == "Softmax") {
            int axis = (opsetVersion >= 13) ? -1 : 1;

            if ((itAttr = attribute.find("axis")) != attribute.end())
                axis = (*itAttr).second->i();

            std::vector<int> perm;
            std::vector<int> invPerm;

            if (axis != 1) {
                // Check if a permutation is need before and after the softmax
                if (axis < 0)
                    axis = (axis + 4);
                axis = 3 - axis;

                perm = {(axis + 1) % 3,
                        (axis + 2) % 3,
                        axis,
                        3};
            }

            if (globTranspose)
                std::swap(perm[0], perm[1]);

            const std::string inputX = redirectName(node.input(0));
            std::shared_ptr<Cell> inputXCell = getCell(inputX);

            const int outputsDim = (!perm.empty()) ? perm[2] : 2;
            const unsigned int nbOutputs = (inputXCell)
                ? inputXCell->getOutputsDim(outputsDim)
                : sp->getSize()[outputsDim];

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            std::shared_ptr<SoftmaxCell> softmaxCell
                = Registrar<SoftmaxCell>::create<Float_T>(model)(*deepNet, 
                                                                node.output(0),
                                                                nbOutputs,
                                                                true,
                                                                0U);

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                throw std::runtime_error("Unsupported operation: Concat before "
                    "Softmax");
            }
            else {
                std::shared_ptr<Cell> inputXCell = getCell(inputX);
                parentCells.push_back(inputXCell);

                if (!perm.empty()) {
                    // Transpose before
                    std::cout << "  Added transpose before: "
                        << perm << std::endl;

                    std::shared_ptr<TransposeCell> transposeBeforeCell
                        = Registrar<TransposeCell>::create<Float_T>(model)(*deepNet, 
                                                                        node.output(0) + "_before",
                                                                        nbOutputs,
                                                                        perm);

                    invPerm = transposeBeforeCell->getInversePermutation();

                    if (inputXCell)
                        transposeBeforeCell->addInput(inputXCell.get());
                    else {
                        transposeBeforeCell->addInput(*sp, 0, 0,
                                            sp->getSizeX(), sp->getSizeY());
                    }

                    deepNet->addCell(transposeBeforeCell, parentCells);
                    transposeBeforeCell->initialize();

                    inputXCell = transposeBeforeCell;
                    parentCells.clear();
                    parentCells.push_back(inputXCell);
                }

                if (inputXCell)
                    softmaxCell->addInput(inputXCell.get());
                else {
                    softmaxCell->addInput(*sp, 0, 0,
                                        sp->getSizeX(), sp->getSizeY());
                }
            }

            deepNet->addCell(softmaxCell, parentCells);
            softmaxCell->initialize();
            cell = softmaxCell;

            if (!invPerm.empty()) {
                // Transpose after
                std::cout << "  Added transpose after: "
                    << invPerm << std::endl;

                const unsigned int nbOutputsAfter
                    = softmaxCell->getOutputsDims()[invPerm[2]];

                std::shared_ptr<TransposeCell> transposeAfterCell
                    = Registrar<TransposeCell>::create<Float_T>(model)(*deepNet, 
                                                                    node.output(0) + "_after",
                                                                    nbOutputsAfter,
                                                                    invPerm);

                transposeAfterCell->addInput(softmaxCell.get());

                deepNet->addCell(transposeAfterCell, parentCells);
                transposeAfterCell->initialize();
                cell = transposeAfterCell;
            }

            std::shared_ptr<Target> target = Registrar
                <Target>::create("TargetScore")(node.output(0) + ".Target",
                                            softmaxCell,
                                            deepNet->getStimuliProvider(),
                                            1.0,
                                            0.0,
                                            1,
                                            "",
                                            false);

            deepNet->addTarget(target);

/*
            // DEBUG
            std::string targetName = Utils::dirName(node.output(0));
            targetName.pop_back();
            targetName = Utils::baseName(targetName);

            if (!targetName.empty()) {
                std::shared_ptr<Target> target = Registrar
                    <Target>::create("TargetCompare")(node.output(0)
                                                                    + ".Target",
                                                softmaxCell,
                                                deepNet->getStimuliProvider(),
                                                1.0,
                                                0.0,
                                                1,
                                                "",
                                                false);
                target->setParameter<std::string>("DataPath",
                                                  "n07745940_14257");
                target->setParameter<std::string>("Matching",
                                                  targetName + ".txt");
                target->setParameter<TargetCompare::TargetFormat>(
                    "TargetFormat", TargetCompare::NHWC);
                target->setParameter<bool>("LogError", true);
                target->setParameter<unsigned int>("BatchPacked", 1);

                deepNet->addTarget(target);
            }
*/
        }
        else if (node.op_type() == "Softplus") {
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (cellFrame->getActivation()
                && cellFrame->getActivation()->getType()
                    != LinearActivation::Type)
            {
                throw std::runtime_error("Cell " + cell->getName()
                    + " already has an activation!");
            }

            cellFrame->setActivation(Registrar<SoftplusActivation>
                ::create<Float_T>(model)());

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //Softsign
        //SpaceToDepth
        //Split
        //SplitToSequence
        //Sqrt
        //Squeeze
        else if (node.op_type() == "Squeeze") {
            std::cout << Utils::cnotice << "  Ignore Squeeze operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //StringNormalizer
        //Sub -> see Sum
        else if (node.op_type() == "Sum" || node.op_type() == "Add"
            || node.op_type() == "Sub" || node.op_type() == "Max"
            || node.op_type() == "Mul" || node.op_type() == "Div")
        {
            const std::string inputData1 = redirectName(node.input(0));
            const std::string inputData2 = redirectName(node.input(1));

            std::string inputData;

            if ((itInit = initializer.find(inputData1)) != initializer.end()
                && initializer.find(inputData2) == initializer.end())
            {
                inputData = inputData2;
            }
            else if ((itInit = initializer.find(inputData2))
                    != initializer.end()
                && initializer.find(inputData1) == initializer.end())
            {
                inputData = inputData1;
            }

            // One of the input is constant, cannot use ElemWiseCell
            if (!inputData.empty() && node.input_size() == 2) {

                std::shared_ptr<Cell> dataCell = getCell(inputData);

                // Checking if this layer adds a constant input full of 0.
                if (node.op_type() == "Add"){
                    Tensor<Float_T> constant = ONNX_unpackTensor<Float_T>((*itInit).second);
                    bool removable = true;
                    for(Tensor<Float_T>::iterator constantIterator = constant.begin(); 
                        constantIterator != constant.end(); 
                        constantIterator++){
                        if ((*constantIterator) != 0){
                            removable = false;
                            break;
                        }
                    }
                    if (removable){
                        std::cout << "  " << node.output(0) << " -> "
                            << inputData << std::endl;
                        redirect[node.output(0)] = inputData;
                        continue;
                    }
                }
                // Special case for bias (CNTK)
                // In CNTK models, bias is added as constant after the operator
                // In this case, we try to merge everything in the operator bias
                if (globCNTK && dataCell
                    && (node.op_type() == "Add" || node.op_type() == "Sum"
                        || (node.op_type() == "Sub" && inputData == inputData1))
                    && (dataCell->getType() == ConvCell::Type
                        || dataCell->getType() == FcCell::Type)
                    && dataCell->getParameter<bool>("NoBias"))
                {
                    // Infer bias from Conv/Fc (without activation) + Add
                    std::shared_ptr<Cell_Frame_Top> dataCellFrame
                        = std::dynamic_pointer_cast<Cell_Frame_Top>(dataCell);

                    if (!dataCellFrame->getActivation()
                        || dataCellFrame->getActivation()->getType()
                            == LinearActivation::Type)
                    {
                        dataCell->setParameter<bool>("NoBias", false);
                        dataCell->initialize(); // Re-init with bias!
                        Tensor<Float_T> biases;

                        std::vector<unsigned int> biasDims;
                        biasDims.push_back(1);
                        biasDims.push_back(dataCell->getNbOutputs());

                        if ((*itInit).second->data_type() == onnx::TensorProto_DataType_FLOAT) {
                            biases.resize({dataCell->getNbOutputs()});
                            biases = ONNX_unpackTensor<Float_T>((*itInit).second);
                        }
                        else if ((*itInit).second->data_type() == onnx::TensorProto_DataType_INT32) {
                            std::cout << Utils::cnotice 
                                << "  Implicit Cast on Biases from INT32 to Float32 for " 
                                << node.op_type()
                                << " for layer type "
                                << Utils::cdef << std::endl;
                            Tensor<int32_t> bTmp = ONNX_unpackTensor<int32_t>((*itInit).second);
                            biases.resize({1, dataCell->getNbOutputs()});
                            for(unsigned int i = 0; i < bTmp.size(); ++i) {
                                biases(i) = (float) bTmp(i);
                            }
                        }
                        else if ((*itInit).second->data_type() == onnx::TensorProto_DataType_DOUBLE) {
                            std::cout << Utils::cnotice 
                                << "  Implicit Cast on Biases from Float64 to Float32 for " 
                                << node.op_type()
                                << " for layer type "
                                << Utils::cdef << std::endl;
                            Tensor<double> bTmp = ONNX_unpackTensor<double>((*itInit).second);
                            biases.resize({1, dataCell->getNbOutputs()});
                            for(unsigned int i = 0; i < bTmp.size(); ++i) {
                                biases(i) = (float) bTmp(i);
                            }
                        }
                        else {
                            throw std::runtime_error("Unsupported datatype: "
                                "Add or Sum  Layer only support Float32, Float64 or INT32 Weights");
                        }

                        if(biases.nbDims() == 1){
                            biases.reshape({1, biases.dimB()});
                        }

                        for (unsigned int output = 0;
                            output < dataCell->getNbOutputs(); ++output)
                        {
                            if (node.op_type() == "Sub")
                                biases[output](0) = -biases[output](0);

                            if (dataCell->getType() == ConvCell::Type) {
                                std::dynamic_pointer_cast<ConvCell>(dataCell)
                                    ->setBias(output, biases[output]);
                            }
                            else if (dataCell->getType() == FcCell::Type) {
                                std::dynamic_pointer_cast<FcCell>(dataCell)
                                    ->setBias(output, biases[output]);
                            }
                        }

                        std::cout << "  " << node.output(0) << " -> "
                            << inputData << std::endl;
                        redirect[node.output(0)] = inputData;
                        continue;
                    }
                }
                else if (dataCell
                    && (node.op_type() == "Add" || node.op_type() == "Sum"
                        || (node.op_type() == "Sub" && inputData == inputData1))
                    && (dataCell->getType() == ScalingCell::Type))
                {
                    // Infer batchnorm from Scaling (Mul) + Add
                    std::shared_ptr<Activation> activation
                        = std::shared_ptr<Activation>();

                    std::shared_ptr<BatchNormCell> batchNormCell
                        = Registrar<BatchNormCell>::create<Float_T>(model)(
                            *deepNet, 
                            node.output(0),
                            dataCell->getNbOutputs(),
                            activation);

                    const std::vector<std::shared_ptr<Cell> > parentCells
                        = dataCell->getParentsCells();
                    const Scaling& scaling
                        = std::dynamic_pointer_cast<ScalingCell>(dataCell)
                            ->getScaling();
                    const auto& scales = scaling.getFloatingPointScaling()
                                                        .getScalingPerOutput();
                    // Remove original "Mul" scaling cell
                    deepNet->removeCell(dataCell);

                    for (auto parentCell: parentCells)
                        batchNormCell->addInput(parentCell.get());

                    deepNet->addCell(batchNormCell, parentCells);
                    batchNormCell->initialize();
                    cell = batchNormCell;

                    std::shared_ptr<Cell_Frame_Top> cellFrame
                        = std::dynamic_pointer_cast<Cell_Frame_Top>(batchNormCell);

                    // Set parameters
                    Tensor<Float_T> biases
                        = ONNX_unpackTensor<Float_T>((*itInit).second);
                    biases.reshape({1, dataCell->getNbOutputs()});

                    if (cellFrame)
                        cellFrame->keepInSync(false);

                    for (unsigned int output = 0;
                        output < batchNormCell->getNbOutputs(); ++output)
                    {
                        const Tensor<Float_T> scale({1}, scales[output]);
                        const Tensor<Float_T> mean({1}, 0.0);
                        const Tensor<Float_T> variance({1}, 1.0
                            - batchNormCell->getParameter<double>("Epsilon"));

                        batchNormCell->setBias(output, biases[output]);
                        batchNormCell->setScale(output, scale);
                        batchNormCell->setMean(output, mean);
                        batchNormCell->setVariance(output, variance);
                    }

                    if (cellFrame)
                        cellFrame->synchronizeToD(true);

                    continue;
                }
                else if (node.op_type() != "Max") {
                    if (inputData == inputData2
                        && (node.op_type() == "Sub" || node.op_type() == "Div"))
                    {
                        // Non-associative operator
                        throw std::runtime_error("Unsupported operation: Sub or"
                            " Div with first operand constant");
                    }

                    const unsigned int nbOutputs = (cell)
                        ? cell->getNbOutputs()
                        : sp->getNbChannels();

                    Tensor<Float_T> constant
                        = ONNX_unpackTensor<Float_T>((*itInit).second);
                    std::shared_ptr<Cell> opCell;
                    std::shared_ptr<Transformation> trans;

                    std::vector<Float_T> weights;
                    std::vector<Float_T> shifts;
                    const ElemWiseCell::Operation operation = ElemWiseCell::Sum;

                    //IRv2 - replace scaling cell by ElWise to be able to train
                    /*
                    if (node.op_type() == "Mul"
                        && (constant.size() == 1
                            || constant.size() == nbOutputs))
                    {
                        // Use ScalingCell for Mul
                        const std::vector<Float_T> scaling
                            = (constant.size() == 1)
                                ? std::vector<Float_T>(nbOutputs, constant(0))
                                : constant.data();

                        opCell = Registrar<ScalingCell>::create<Float_T>(model)(
                            *deepNet, 
                            node.output(0),
                            nbOutputs,
                            Scaling::floatingPointScaling(scaling, false, std::vector<Float_T>(0.0f)));

                        if (constant.size() == 1) {
                            std::cout << "  scaling factor = " << constant(0)
                                << std::endl;
                        }
                    }
                    */
                    if (constant.size() == 1) {
                        if(node.op_type() == "Add"){
                            std::cout << "Add operation" << std::endl;
                            shifts.push_back(constant(0));
                            weights.push_back(1);
                        }
                        else if(node.op_type() == "Div"){
                            std::cout << "Div operation" << std::endl;
                            shifts.push_back(0);
                            weights.push_back(1./constant(0));
                        }
                        else if(node.op_type() == "Mul"){
                            std::cout << "Mul operation" << std::endl;
                            shifts.push_back(0);
                            weights.push_back(constant(0));
                        }
                    }
                    else if (constant.size() == nbOutputs){
                        if(node.op_type() == "Add"){
                            std::cout << "Add operation, constant.size() == nbOutputs" << std::endl;
                            for (unsigned int output = 0;
                                    output < nbOutputs; ++output)
                            {
                                shifts.push_back(constant.at(output));
                                weights.push_back(1);
                                std::cout << "out = " << output << " , shift = " << constant.at(output) << std::endl;
                            }
                        }
                    }
                    else {
                        throw std::runtime_error("Unsupported constant size! Not 1 and not nbOutputs! ");
                    }

                    std::map<std::string, std::vector<std::string> >
                        ::const_iterator itConcat;
                    std::vector<std::shared_ptr<Cell> > parentCells;

                    if (!opCell) {
                        const ElemWiseCell::CoeffMode coeffMode = ElemWiseCell::PerLayer;

                        std::shared_ptr<Activation> activation
                                = std::shared_ptr<Activation>();

                        opCell = Registrar<ElemWiseCell>::create(model)(deepNet->getNetwork(),
                                                            *deepNet,
                                                            node.output(0),
                                                            nbOutputs,
                                                            operation,
                                                            coeffMode,
                                                            weights,
                                                            shifts,
                                                            activation);
                    }

                    if ((itConcat = concat.find(inputData)) != concat.end()) {
                        throw std::runtime_error("Unsupported operation: Concat before "
                            "Add, Sum, Sub, Mul or Div");
                    }
                    else {
                        parentCells.push_back(cell);

                        if (cell)
                            opCell->addInput(cell.get());
                        else {
                            opCell->addInput(*sp, 0, 0,
                                                sp->getSizeX(), sp->getSizeY());
                        }
                    }

                    deepNet->addCell(opCell, parentCells);
                    opCell->initialize();
                    cell = opCell;
                    continue;
                }
            }

            if (node.op_type() == "Div") {
                throw std::runtime_error("Unsupported operation: Div with both"
                    " operands non-constant");
            }

            std::shared_ptr<Cell> inputDataCell = getCell(inputData1);

            const ElemWiseCell::Operation operation
                = (node.op_type() == "Max") ? ElemWiseCell::Max
                : (node.op_type() == "Mul") ? ElemWiseCell::Prod
                : ElemWiseCell::Sum;
            std::vector<Float_T> weights;
            std::vector<Float_T> shifts;

            if (node.op_type() == "Sub") {
                weights.push_back(1.0);
                weights.push_back(-1.0);
            }
            const ElemWiseCell::CoeffMode coeffMode = ElemWiseCell::PerLayer;

            std::shared_ptr<Activation> activation
                = std::shared_ptr<Activation>();

            std::shared_ptr<ElemWiseCell> elemWiseCell
                = Registrar<ElemWiseCell>::create(model)(deepNet->getNetwork(),
                                                            *deepNet, 
                                                            node.output(0),
                                                            inputDataCell->getNbOutputs(),
                                                            operation,
                                                            coeffMode,
                                                            weights,
                                                            shifts,
                                                            activation);

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            for (int i = 0; i < node.input_size(); ++i) {
                const std::string inputData = redirectName(node.input(i));

                if ((itConcat = concat.find(inputData)) != concat.end()) {
                    throw std::runtime_error("Unsupported operation: Concat before "
                        "ElemWise");
                }
                else {
                    std::shared_ptr<Cell> inputDataCell = getCell(inputData);
                    parentCells.push_back(inputDataCell);

                    if (inputDataCell)
                        elemWiseCell->addInput(inputDataCell.get());
                    else {
                        elemWiseCell->addInput(*sp, 0, 0,
                                            sp->getSizeX(), sp->getSizeY());
                    }
                }
            }

            deepNet->addCell(elemWiseCell, parentCells);
            elemWiseCell->initialize();
            cell = elemWiseCell;
        }
        //Tan
        else if (node.op_type() == "Tanh") {
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (cellFrame->getActivation()
                && cellFrame->getActivation()->getType()
                    != LinearActivation::Type)
            {
                throw std::runtime_error("Cell " + cell->getName()
                    + " already has an activation!");
            }

            cellFrame->setActivation(Registrar<TanhActivation>
                ::create<Float_T>(model)());

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //TfIdfVectorizer
        //ThresholdedRelu
        //Tile
        //TopK
        else if (node.op_type() == "Transpose") {
            const std::string inputX = redirectName(node.input(0));
            std::shared_ptr<Cell> inputXCell = getCell(inputX);

            // perm
            std::vector<int> perm;

            if ((itAttr = attribute.find("perm")) != attribute.end()) {
                const int lastDim = (*itAttr).second->ints_size() - 1;

                for (int dim = lastDim; dim >= 0; --dim)
                    perm.push_back(lastDim - (*itAttr).second->ints(dim));
            }
            else {
                // By default, reverse the dimensions.
                perm.resize(4);
                perm[0] = 3;
                perm[1] = 2;
                perm[2] = 1;
                perm[3] = 0;
            }
            const std::vector<size_t>& outputDims = (inputXCell)
                ? inputXCell->getOutputsDims()
                : sp->getSize();
            int nbDimEqualOne = 0;
            for (size_t outputDim : outputDims)
                nbDimEqualOne += (outputDim == 1) ? 1 : 0;
            
            const unsigned int nbOutputs = outputDims[perm[2]];
            std::map<std::string, std::vector<std::string> >
                    ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            if (globTranspose)
                std::swap(perm[0], perm[1]);
            
            if (nbDimEqualOne == 2){
                // Only one dimension (except batch size) is nonunary
                // This transpose is also a reshape which doesn't change the memory layout
                if (perm[3] != 3){
                    throw std::domain_error("TransposeCell: "
                                "permutation of the fourth (batch) dimension "
                                "is not supported.");
                }
                std::vector<int> dims = {};
                for (int permIdx=0; permIdx<2; ++permIdx) 
                    dims.push_back(outputDims[perm[permIdx]]);
                dims.push_back(-1); // Batchsize doesn't change
                std::shared_ptr<ReshapeCell> reshapeCell
                    = Registrar<ReshapeCell>::create<Float_T>(model)(*deepNet, 
                                                                    node.output(0),
                                                                    nbOutputs,
                                                                    dims);
                if ((itConcat = concat.find(inputX)) != concat.end()) {
                    throw std::runtime_error("Unsupported operation: Concat before "
                        "Transpose");
                } else {
                    std::shared_ptr<Cell> inputXCell = getCell(inputX);
                    parentCells.push_back(inputXCell);

                    if (inputXCell)
                        reshapeCell->addInput(inputXCell.get());
                    else {
                        reshapeCell->addInput(*sp, 0, 0,
                                            sp->getSizeX(), sp->getSizeY());
                    }
                }

                deepNet->addCell(reshapeCell, parentCells);
                reshapeCell->initialize();
                cell = reshapeCell;
            }else{
                // Normal case, we add the Transpose layer
                std::shared_ptr<TransposeCell> transposeCell
                    = Registrar<TransposeCell>::create<Float_T>(model)(*deepNet, 
                                                                    node.output(0),
                                                                    nbOutputs,
                                                                    perm);
                if ((itConcat = concat.find(inputX)) != concat.end()) {
                throw std::runtime_error("Unsupported operation: Concat before "
                    "Transpose");
                }
                else {
                    std::shared_ptr<Cell> inputXCell = getCell(inputX);
                    parentCells.push_back(inputXCell);

                    if (inputXCell)
                        transposeCell->addInput(inputXCell.get());
                    else {
                        transposeCell->addInput(*sp, 0, 0,
                                            sp->getSizeX(), sp->getSizeY());
                    }
                }

                deepNet->addCell(transposeCell, parentCells);
                transposeCell->initialize();
                cell = transposeCell;
            }
        }
        //Unique
        //Unsqueeze
        else if (node.op_type() == "Unsqueeze") {
            assert(node.attribute_size() > 0);

            std::cout << Utils::cnotice << "  Ignore Unsqueeze operation to "
                << node.attribute(0).GetTypeName()
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //else if (node.op_type() == "Upsample") {

        //}
        //Where
        //Xor
        else {
            throw std::runtime_error("Unsupported ONNX operator: "
                                     + node.op_type());
        }

        if (cell) {
            std::cout << "  # Inputs dims: "
                << cell->getInputsDims() << std::endl;
            std::cout << "  # Outputs dims: "
                << cell->getOutputsDims() << std::endl;
        }
    }
/*
    if (deepNet->getTargets().empty()) {
        std::cout << Utils::cnotice << "No target specified, adding default "
            "target to the last layer: " << cell->getName() << Utils::cdef
            << std::endl;

        std::shared_ptr<Target> target = Registrar
            <Target>::create("TargetScore")(cell->getName() + ".Target",
                                        cell,
                                        deepNet->getStimuliProvider(),
                                        1.0,
                                        0.0,
                                        1,
                                        "",
                                        false);

        deepNet->addTarget(target);
    }
*/
}
#endif
