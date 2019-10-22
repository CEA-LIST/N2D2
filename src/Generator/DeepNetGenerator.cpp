/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Johannes THIELE (olivier.bichler@cea.fr)

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
#include "Monitor.hpp"
#include "Network.hpp"
#include "NodeEnv.hpp"
#include "StimuliProvider.hpp"
#include "Synapse_PCM.hpp"
#include "Synapse_RRAM.hpp"
#include "Synapse_Static.hpp"
#include "Cell/Cell_CSpike.hpp"
#include "Cell/Cell_Spike.hpp"
#include "Cell/NodeIn.hpp"
#include "Cell/NodeOut.hpp"
#include "Generator/CellGenerator.hpp"
#include "Generator/CEnvironmentGenerator.hpp"
#include "Generator/DatabaseGenerator.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "Generator/EnvironmentGenerator.hpp"
#include "Generator/TargetGenerator.hpp"

#ifdef ONNX
#include "N2D2.hpp"
#include "Database/ILSVRC2012_Database.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "Transformation/PadCropTransformation.hpp"
#include "Transformation/ColorSpaceTransformation.hpp"
#include "Transformation/RangeAffineTransformation.hpp"
#include "Cell/BatchNormCell.hpp"
#include "Cell/ConvCell.hpp"
#include "Cell/DropoutCell.hpp"
#include "Cell/ElemWiseCell.hpp"
#include "Cell/LRNCell.hpp"
#include "Cell/PaddingCell.hpp"
#include "Cell/PoolCell.hpp"
#include "Cell/SoftmaxCell.hpp"

#include "third_party/onnx/onnx.proto3.pb.hpp"
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
    else if (fileExtension == "onnx")
        return generateFromONNX(network, fileName);
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
    deepNet->setParameter("SignalsDiscretization",
        iniConfig.getProperty<unsigned int>("SignalsDiscretization", 0U));
    deepNet->setParameter("FreeParametersDiscretization",
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

            for (std::vector<std::string>::const_iterator itParent
                 = (*itParents).second.begin();
                 itParent != (*itParents).second.end();
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

            std::shared_ptr<Cell_CSpike> cellCSpike = std::dynamic_pointer_cast
                <Cell_CSpike>(cell);
             // Monitor for the cell
            // Try different casts to find out Cell type
            std::shared_ptr<Cell_Spike> cellSpike = std::dynamic_pointer_cast
                <Cell_Spike>(cell);

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
                std::cout << "Warning: No monitor could be added to Cell: " +
                    cell->getName() << std::endl;
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

    std::shared_ptr<Environment> env = std::dynamic_pointer_cast<Environment>
        (deepNet->getStimuliProvider());

    std::shared_ptr<CEnvironment> Cenv = std::dynamic_pointer_cast<CEnvironment>
        (deepNet->getStimuliProvider());

    if (Cenv) {
#ifdef CUDA
        std::shared_ptr<CMonitor> cmonitor(new CMonitor_CUDA());
        cmonitor->add(Cenv->getTickOutputs());

        deepNet->addCMonitor("env", cmonitor);

#else
        std::shared_ptr<CMonitor> cmonitor(new CMonitor());
        cmonitor->add(Cenv->getTickOutputs());

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
                                         const std::string& fileName)
{
    std::shared_ptr<DeepNet> deepNet(new DeepNet(network));
    deepNet->setParameter("Name", Utils::baseName(fileName));

    //std::cout << Utils::cwarning << "Warning: no database specified."
    //            << Utils::cdef << std::endl;
    //deepNet->setDatabase(std::make_shared<Database>());
    std::shared_ptr<ILSVRC2012_Database> database = std::make_shared
        <ILSVRC2012_Database>(1.0, true, false);
    database->load(N2D2_DATA("ILSVRC2012"),
                   N2D2_DATA("ILSVRC2012/synsets.txt"));
    deepNet->setDatabase(database);

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    onnx::ModelProto onnxModel;

    std::ifstream onnxFile(fileName.c_str(), std::ios::binary);

    if (!onnxFile.good())
        throw std::runtime_error("Could not open ONNX file: " + fileName);
    else if (!onnxModel.ParseFromIstream(&onnxFile))
        throw std::runtime_error("Failed to parse ONNX file: " + fileName);

    onnxFile.close();

    std::cout << "Importing ONNX model:\n"
        "  ir_version = " << onnxModel.ir_version() << "\n"
        "  producer_name = " << onnxModel.producer_name() << "\n"
        "  producer_version = " << onnxModel.producer_version() << "\n"
        "  domain = " << onnxModel.domain() << "\n"
        "  model_version = " << onnxModel.model_version() << "\n"
        "  doc_string = " << onnxModel.doc_string() << std::endl;

    ONNX_processGraph(deepNet, onnxModel.graph());

    return deepNet;
}

void N2D2::DeepNetGenerator::ONNX_processGraph(std::shared_ptr<DeepNet> deepNet,
                                              const onnx::GraphProto& graph)
{
#ifdef CUDA
    const std::string model = "Frame_CUDA";
#else
    const std::string model = "Frame";
#endif

    std::map<std::string, const onnx::TensorProto*> initializer;
    for (int i = 0; i < graph.initializer_size(); ++i) {
        const onnx::TensorProto* tensor = &(graph.initializer(i));
        initializer[tensor->name()] = tensor;
    }

    std::map<std::string, const onnx::ValueInfoProto*> input;
    std::map<std::string, const onnx::ValueInfoProto*> dataInput;
    for (int i = 0; i < graph.input_size(); ++i) {
        const onnx::ValueInfoProto* valueInfo = &(graph.input(i));
        input[valueInfo->name()] = valueInfo;

        if (initializer.find(valueInfo->name()) == initializer.end())
            dataInput[valueInfo->name()] = valueInfo;
    }

    std::map<std::string, const onnx::ValueInfoProto*> output;
    for (int o = 0; o < graph.output_size(); ++o) {
        const onnx::ValueInfoProto* valueInfo = &(graph.output(o));
        output[valueInfo->name()] = valueInfo;
    }

    // Input: StimuliProvider construction
    if (dataInput.size() != 1)
        throw std::runtime_error("Number of data input should be 1");

    const onnx::TypeProto_Tensor& inputType
        = (*dataInput.begin()).second->type().tensor_type();
    const onnx::TensorShapeProto& shape = inputType.shape();

    std::vector<size_t> size;
    for (int i = 1; i < shape.dim_size(); ++i)
        size.push_back(shape.dim(i).dim_value());
    std::reverse(size.begin(), size.end());

    unsigned int batchSize = shape.dim(0).dim_value();
    if (batchSize < 1)
        batchSize = 1;

    const bool compositeStimuli = false;

    std::shared_ptr<StimuliProvider> sp(new StimuliProvider(
        *deepNet->getDatabase(), size, batchSize, compositeStimuli));
    deepNet->setStimuliProvider(sp);

    std::cout << "StimuliProvider: " << size << " (" << batchSize << ")"
        << std::endl;

    // Pre-processing for ImageNet used by the MobileNet families
    RescaleTransformation rescale(256, 256);
    rescale.setParameter<bool>("KeepAspectRatio", true);
    rescale.setParameter<bool>("ResizeToFit", false);

    sp->addTransformation(rescale);
    sp->addTransformation(PadCropTransformation(size[0], size[1]));
    sp->addTransformation(ColorSpaceTransformation(
        ColorSpaceTransformation::BGR));
    sp->addTransformation(RangeAffineTransformation(
        RangeAffineTransformation::Minus, {127.5},
        RangeAffineTransformation::Divides, {127.5}));

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

    for (int n = 0; n < graph.node_size(); ++n) {
        const onnx::NodeProto& node = graph.node(n);

        std::cout << "Layer: " << node.name() << " [" << node.op_type() << "]"
            << std::endl;
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
        std::map<std::string, const onnx::TensorProto*>
            ::const_iterator itInit;

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
            || node.op_type() == "MaxPool"
            || node.op_type() == "GlobalMaxPool")
        {
            const std::string inputX = redirectName(node.input(0));
            std::shared_ptr<Cell> inputXCell
                = (deepNet->getCells().empty())
                    ? std::shared_ptr<Cell>()
                    : deepNet->getCell(inputX);

            // kernel_shape
            std::vector<unsigned int> kernelDims;

            if ((itAttr = attribute.find("kernel_shape")) != attribute.end()) {
                for (int dim = 0; dim < (*itAttr).second->ints_size(); ++dim)
                    kernelDims.push_back((*itAttr).second->ints(dim));

                std::reverse(kernelDims.begin(), kernelDims.end());
            }
            else if (node.op_type() == "GlobalAveragePool"
                || node.op_type() == "GlobalMaxPool")
            {
                assert(inputXCell);

                const std::vector<size_t>& inputsDims
                    = inputXCell->getOutputsDims();

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
                    const int padding = (kernelDims[dim] + strideDims[dim]);
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
                if ((*itAttr).second->i() != 0) {
                    std::cout << Utils::cwarning << "Unsupported operation: "
                        << node.op_type() << " with ceil_mode != 0"
                        << Utils::cdef << std::endl;
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

            const std::vector<unsigned int> paddingDims = (paddingCellRequired)
                ? std::vector<unsigned int>(kernelDims.size(), 0U)
                : paddingDimsBegin;

            // Make a unit map
            Tensor<bool> map({inputXCell->getNbOutputs(),
                              inputXCell->getNbOutputs()}, false);

            for (unsigned int i = 0; i < inputXCell->getNbOutputs(); ++i)
                map(i, i) = true;

            const PoolCell::Pooling pooling = (node.op_type() == "AveragePool"
                || node.op_type() == "GlobalAveragePool")
                    ? PoolCell::Average : PoolCell::Max;

            std::shared_ptr<PoolCell> poolCell
                = Registrar<PoolCell>::create<Float_T>(model)(deepNet->getNetwork(),
                                                                *deepNet, 
                                                                node.output(0),
                                                                kernelDims,
                                                                inputXCell->getNbOutputs(),
                                                                strideDims,
                                                                paddingDims,
                                                                pooling,
                                                                activation);

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            if (paddingCellRequired) {
                std::cout << "  Added padding: " << paddingDimsBegin
                    << " -- " << paddingDimsEnd << std::endl;

                std::shared_ptr<PaddingCell> paddingCell = Registrar
                    <PaddingCell>::create(model)(*deepNet,
                                                node.output(0) + "_padding",
                                                inputXCell->getNbOutputs(),
                                                paddingDimsBegin[1],
                                                paddingDimsEnd[1],
                                                paddingDimsBegin[0],
                                                paddingDimsEnd[0]);

                if ((itConcat = concat.find(inputX)) != concat.end()) {
                    throw std::runtime_error("Unsupported operation: "
                        "Concat before Pool");
                }
                else {
                    std::shared_ptr<Cell> inputXCell
                        = (deepNet->getCells().empty())
                            ? std::shared_ptr<Cell>()
                            : deepNet->getCell(inputX);
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
                    throw std::runtime_error("Unsupported operation: "
                        "Concat before Pool");
                }
                else {
                    std::shared_ptr<Cell> inputXCell
                        = (deepNet->getCells().empty())
                            ? std::shared_ptr<Cell>()
                            : deepNet->getCell(inputX);
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
        }
        else if (node.op_type() == "BatchNormalization") {
            const std::string inputScale = node.input(1);
            const unsigned int nbOutputs = initializer[inputScale]->dims(0);

            std::shared_ptr<Activation> activation
                = std::shared_ptr<Activation>();

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

            const std::string inputX = redirectName(node.input(0));
            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                throw std::runtime_error("Unsupported operation: Concat before "
                    "BatchNorm");
            }
            else {
                std::shared_ptr<Cell> inputXCell
                    = (deepNet->getCells().empty())
                        ? std::shared_ptr<Cell>()
                        : deepNet->getCell(inputX);
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
        //BitShift
        else if (node.op_type() == "Cast") {
            assert(node.attribute_size() > 0);

            std::cout << Utils::cnotice << "  Ignore Cast operation to "
                << onnx::TensorProto_DataType_Name(
                    (onnx::TensorProto_DataType) node.attribute(0).i())
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //Ceil
        else if (node.op_type() == "Clip") {
            Float_T minVal = std::numeric_limits<Float_T>::lowest();
            Float_T maxVal = std::numeric_limits<Float_T>::max();

            if ((itAttr = attribute.find("min")) != attribute.end())
                minVal = (*itAttr).second->f();

            if ((itAttr = attribute.find("max")) != attribute.end())
                maxVal = (*itAttr).second->f();

            if (minVal == 0.0 && maxVal > 0.0) {
                std::shared_ptr<Activation> activation
                    = Registrar<RectifierActivation>::create<Float_T>(model)();

                if (maxVal != std::numeric_limits<Float_T>::max())
                    activation->setParameter<double>("Clipping", maxVal);

                std::shared_ptr<Cell_Frame_Top> cellFrame
                    = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

                if (cellFrame->getActivation()) {
                    throw std::runtime_error("Cell " + cell->getName()
                        + " already has an activation!");
                }

                cellFrame->setActivation(activation);
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
            bool newInsert;
            std::map<std::string, std::vector<std::string> >::iterator it;
            std::tie(it, newInsert) = concat.insert(
                std::make_pair(node.output(0), std::vector<std::string>()));
            assert(newInsert);

            for (int i = 0; i < node.input_size(); ++i) {
                const std::string input = redirectName(node.input(i));
                (*it).second.push_back(input);
            }
        }
        //ConcatFromSequence
        //Constant
        //ConstantOfShape
        else if (node.op_type() == "Conv") {
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
                    const int padding = (kernelExtent + strideDims[dim]);
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

            const std::vector<unsigned int> subSampleDims(kernelDims.size(), 1);
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

            const std::vector<int> paddingDims = (paddingCellRequired)
                ? std::vector<int>(kernelDims.size(), 0) : paddingDimsBegin;

            // Cell construction
            std::shared_ptr<ConvCell> convCell
                = Registrar<ConvCell>::create<Float_T>(model)(deepNet->getNetwork(),
                                                                *deepNet, 
                                                                node.output(0),
                                                                kernelDims,
                                                                nbOutputs,
                                                                subSampleDims,
                                                                strideDims,
                                                                paddingDims,
                                                                dilationDims,
                                                                activation);

            // Parameters
            convCell->setParameter<bool>("NoBias", (node.input_size() != 3));

            const std::string inputX = redirectName(node.input(0));
            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            if (paddingCellRequired) {
                std::cout << "  Added padding: " << paddingDimsBegin
                    << " -- " << paddingDimsEnd << std::endl;

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
                        std::shared_ptr<Cell> inputCell = deepNet->getCell(input);
                        parentCells.push_back(inputCell);

                        paddingCell->addInput(inputCell.get());
                    }
                }
                else {
                    std::shared_ptr<Cell> inputXCell
                        = (deepNet->getCells().empty())
                            ? std::shared_ptr<Cell>()
                            : deepNet->getCell(inputX);
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
                        std::shared_ptr<Cell> inputCell = deepNet->getCell(input);
                        parentCells.push_back(inputCell);

                        convCell->addInput(inputCell.get(), map);
                    }
                }
                else {
                    std::shared_ptr<Cell> inputXCell
                        = (deepNet->getCells().empty())
                            ? std::shared_ptr<Cell>()
                            : deepNet->getCell(inputX);
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
            cell = convCell;

            std::cout << "  # Shared synapses: "
                << convCell->getNbSharedSynapses() << std::endl;
            std::cout << "  # Virtual synapses: "
                << convCell->getNbVirtualSynapses() << std::endl;

            convCell->writeMap("map/" + node.output(0) + "_map.dat");

            // Free parameters
            if (node.input_size() > 1
                && (itInit = initializer.find(node.input(1)))
                    != initializer.end())
            {
                kernelDims.push_back(nbInputs);
                kernelDims.push_back(convCell->getNbOutputs());

                const Tensor<Float_T> kernels
                    = ONNX_unpackTensor<Float_T>((*itInit).second, kernelDims);

                for (unsigned int output = 0;
                    output < convCell->getNbOutputs(); ++output)
                {
                    for (unsigned int channel = 0;
                        channel < convCell->getNbChannels(); ++channel)
                    {
                        if (!convCell->isConnection(channel, output))
                            continue;

                        convCell->setWeight(output, channel,
                                            kernels[output][channel / group]);
                    }
                }
            }
            else if (node.input_size() > 1) {
                std::cout << "  No initializer for \"" << node.input(1)
                    << "\"" << std::endl;
            }

            if (node.input_size() > 2
                && (itInit = initializer.find(node.input(2)))
                    != initializer.end())
            {
                Tensor<Float_T> biases = ONNX_unpackTensor<Float_T>(
                    (*itInit).second, {(unsigned int)convCell->getNbOutputs()});
                biases.reshape({1, convCell->getNbOutputs()});

                for (unsigned int output = 0;
                    output < convCell->getNbOutputs(); ++output)
                {
                    convCell->setBias(output, biases[output]);
                }
            }
            else if (node.input_size() > 2) {
                std::cout << "  No initializer for \"" << node.input(2)
                    << "\"" << std::endl;
            }
        }
        //ConvInteger
        //else if (node.op_type() == "ConvTranspose") {

        //}
        //Cos
        //Cosh
        //CumSum
        //DepthToSpace
        //DequantizeLinear
        //Det
        //Div
        else if (node.op_type() == "Dropout") {
            float ratio = 0.5;

            if ((itAttr = attribute.find("ratio")) != attribute.end())
                ratio = (*itAttr).second->f();

            const std::string inputX = redirectName(node.input(0));
            std::shared_ptr<Cell> inputXCell
                = (deepNet->getCells().empty())
                    ? std::shared_ptr<Cell>()
                    : deepNet->getCell(inputX);

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            std::shared_ptr<DropoutCell> dropoutCell
                = Registrar<DropoutCell>::create<Float_T>(model)(*deepNet, 
                                                                node.output(0),
                                                                inputXCell->getNbOutputs());

            dropoutCell->setParameter<double>("Dropout", ratio);

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                throw std::runtime_error("Unsupported operation: Concat before "
                    "Dropout");
            }
            else {
                std::shared_ptr<Cell> inputXCell
                    = (deepNet->getCells().empty())
                        ? std::shared_ptr<Cell>()
                        : deepNet->getCell(inputX);
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
        //GatherElements
        //GatherND
        //else if (node.op_type() == "Gemm") {

        //}
        //GlobalAveragePool -> see AveragePool
        //GlobalLpPool
        //GlobalMaxPool -> see AveragePool
        //Greater
        //HardSigmoid
        //Hardmax
        //Identity
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
            std::shared_ptr<Cell> inputXCell
                = (deepNet->getCells().empty())
                    ? std::shared_ptr<Cell>()
                    : deepNet->getCell(inputX);

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

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                throw std::runtime_error("Unsupported operation: Concat before "
                    "LRN");
            }
            else {
                std::shared_ptr<Cell> inputXCell
                    = (deepNet->getCells().empty())
                        ? std::shared_ptr<Cell>()
                        : deepNet->getCell(inputX);
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

            if (cellFrame->getActivation()) {
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
        //MatMul
        //MatMulInteger
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
        //else if (node.op_type() == "Pad") {

        //}
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

            if (cellFrame->getActivation()) {
                throw std::runtime_error("Cell " + cell->getName()
                    + " already has an activation!");
            }

            cellFrame->setActivation(Registrar<RectifierActivation>
                ::create<Float_T>(model)());

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        else if (node.op_type() == "Reshape") {
            std::cout << Utils::cnotice << "  Ignore Reshape operation"
                << Utils::cdef << std::endl;

            std::cout << "  " << node.output(0) << " -> "
                << redirectName(node.input(0)) << std::endl;
            redirect[node.output(0)] = redirectName(node.input(0));
            continue;
        }
        //else if (node.op_type() == "Resize") {

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

            if (cellFrame->getActivation()) {
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
        else if (node.op_type() == "Softmax") {
            if ((itAttr = attribute.find("axis")) != attribute.end()) {
                if ((*itAttr).second->i() != 1) {
                    throw std::runtime_error("Unsupported operation: "
                        "Softmax with axis != 1");
                }
            }

            const std::string inputX = redirectName(node.input(0));
            std::shared_ptr<Cell> inputXCell
                = (deepNet->getCells().empty())
                    ? std::shared_ptr<Cell>()
                    : deepNet->getCell(inputX);

            std::map<std::string, std::vector<std::string> >
                ::const_iterator itConcat;
            std::vector<std::shared_ptr<Cell> > parentCells;

            std::shared_ptr<SoftmaxCell> softmaxCell
                = Registrar<SoftmaxCell>::create<Float_T>(model)(*deepNet, 
                                                                node.output(0),
                                                                inputXCell->getNbOutputs(),
                                                                true,
                                                                0U);

            if ((itConcat = concat.find(inputX)) != concat.end()) {
                throw std::runtime_error("Unsupported operation: Concat before "
                    "Softmax");
            }
            else {
                std::shared_ptr<Cell> inputXCell
                    = (deepNet->getCells().empty())
                        ? std::shared_ptr<Cell>()
                        : deepNet->getCell(inputX);
                parentCells.push_back(inputXCell);

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
        }
        else if (node.op_type() == "Softplus") {
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (cellFrame->getActivation()) {
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
            || node.op_type() == "Mul")
        {
            const std::string inputData = redirectName(node.input(0));
            std::shared_ptr<Cell> inputDataCell
                = (deepNet->getCells().empty())
                    ? std::shared_ptr<Cell>()
                    : deepNet->getCell(inputData);

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

            std::shared_ptr<Activation> activation
                = std::shared_ptr<Activation>();

            std::shared_ptr<ElemWiseCell> elemWiseCell
                = Registrar<ElemWiseCell>::create(model)(deepNet->getNetwork(),
                                                            *deepNet, 
                                                            node.output(0),
                                                            inputDataCell->getNbOutputs(),
                                                            operation,
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
                    std::shared_ptr<Cell> inputDataCell
                        = (deepNet->getCells().empty())
                            ? std::shared_ptr<Cell>()
                            : deepNet->getCell(inputData);
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

            if (cellFrame->getActivation()) {
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
        //Transpose
        //Unique
        //Unsqueeze
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
}
#endif


#ifdef PYBIND
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_DeepNetGenerator(py::module &m) {
    py::class_<DeepNetGenerator>(m, "DeepNetGenerator")
    .def_static("generate", &DeepNetGenerator::generate, py::arg("network"), py::arg("fileName"));
}
}
#endif
