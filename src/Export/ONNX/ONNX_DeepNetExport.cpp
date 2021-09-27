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

#ifdef ONNX

#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "DeepNet.hpp"
#include "DrawNet.hpp"
#include "StimuliProvider.hpp"
#include "Cell/ConvCell.hpp"
#include "Cell/FcCell.hpp"
#include "Cell/PoolCell.hpp"
#include "Cell/ElemWiseCell.hpp"
#include "Cell/ScalingCell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Target/TargetScore.hpp"
#include "Export/CellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/ONNX/ONNX_CellExport.hpp"
#include "Export/ONNX/ONNX_DeepNetExport.hpp"
#include "Export/ONNX/Cells/ONNX_ConcatCell.hpp"
#include "utils/IniParser.hpp"
#include "utils/Registrar.hpp"

N2D2::Registrar<N2D2::DeepNetExport>
N2D2::ONNX_DeepNetExport::mRegistrar(
    {"ONNX"},
    N2D2::ONNX_DeepNetExport::generate);

void N2D2::ONNX_DeepNetExport::generate(DeepNet& deepNet,
                                       const std::string& dirName)
{
    Utils::createDirectories(dirName);

    deepNet.fusePadding();  // probably already done, but make sure!
    addBranchesCells(deepNet);

    IniParser exportParams;

    if(!DeepNetExport::mExportParameters.empty())
        exportParams.load(DeepNetExport::mExportParameters);

    DrawNet::drawGraph(deepNet, dirName + "/graph");

    saveModel(deepNet, dirName + "/model.onnx");
}

void N2D2::ONNX_DeepNetExport::saveModel(DeepNet& deepNet,
                                             const std::string& fileName)
{
    const onnx::ModelProto model = generateModel(deepNet);

    std::ofstream onnxFile(fileName.c_str(), std::ios::binary);

    if (!onnxFile.good())
        throw std::runtime_error("Could not create ONNX file: " + fileName);

    google::protobuf::io::OstreamOutputStream zero_copy_output(&onnxFile);

    if (!model.SerializeToZeroCopyStream(&zero_copy_output)) {
        throw std::runtime_error("Error occured during ONNX model "
            "serialization");
    }
}

onnx::ModelProto N2D2::ONNX_DeepNetExport::generateModel(DeepNet& deepNet)
{

    // Create an ONNX model
    onnx::ModelProto model;
    model.set_ir_version(onnx::Version::IR_VERSION);
    model.set_producer_name("N2D2");
    model.set_producer_version("1.0");

    // Create a graph in the ONNX model
    onnx::GraphProto *graph = model.mutable_graph();
    onnx::OperatorSetIdProto *opset = model.add_opset_import();
    opset->set_version(11);

    // Set graph inputs dims
    const std::vector<std::shared_ptr<Cell> > inputCells
        = deepNet.getChildCells("env");

    for (auto itCell = inputCells.begin(); itCell != inputCells.end(); ++itCell)
    {
        std::shared_ptr<Cell_Frame_Top> cellTop = std::dynamic_pointer_cast
            <Cell_Frame_Top>(*itCell);

        onnx::ValueInfoProto *inputInfo = graph->add_input();
        inputInfo->set_name("env");

        setTensorProto(inputInfo, cellTop->getInputs(0));
    }

    // Set graph outputs dims
    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();
    const unsigned int nbTarget = outputTargets.size();

    for (unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        std::shared_ptr<Cell_Frame_Top> cellTop = std::dynamic_pointer_cast
            <Cell_Frame_Top>(cell);

        onnx::ValueInfoProto *inputInfo = graph->add_output();
        inputInfo->set_name(cell->getName());

        setTensorProto(inputInfo, cellTop->getOutputs());
    }

    // Generate graph
    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

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
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

            ONNX_CellExport::getInstance(*cell)->generateNode(
                graph, deepNet, *cell);
        }
    }

    return model;
}

void N2D2::ONNX_DeepNetExport::setTensorProto(
    onnx::ValueInfoProto* info,
    const BaseTensor& tensor)
{

    onnx::TypeProto *inputType = info->mutable_type();
    onnx::TypeProto::Tensor *inputTypeTensor
        = inputType->mutable_tensor_type();
    inputTypeTensor->set_elem_type(getElemType(tensor));

    onnx::TensorShapeProto *inputTypeTensorShape
        = inputTypeTensor->mutable_shape();
    onnx::TensorShapeProto::Dimension *inputTypeTensorDim;
    std::vector<size_t> dims = tensor.dims();
    std::reverse(dims.begin(), dims.end());

    // Set the first dimension to a variable named "batch", 
    // to avoid setting a fixed batch size
    inputTypeTensorDim = inputTypeTensorShape->add_dim();
    inputTypeTensorDim->set_dim_param("batch");

    for (size_t dim = 1; dim < dims.size(); ++dim) {
        inputTypeTensorDim = inputTypeTensorShape->add_dim();
        inputTypeTensorDim->set_dim_value(dims[dim]);
    }
}

onnx::TensorProto::DataType N2D2::ONNX_DeepNetExport::getElemType(
    const BaseTensor& tensor)
{
    if (tensor.getType() == &typeid(float))
        return onnx::TensorProto::FLOAT;
    else if (tensor.getType() == &typeid(half_float::half))
        return onnx::TensorProto::FLOAT16;
    else if (tensor.getType() == &typeid(double))
        return onnx::TensorProto::DOUBLE;
    else if (tensor.getType() == &typeid(int8_t))
        return onnx::TensorProto::INT8;
    else if (tensor.getType() == &typeid(int16_t))
        return onnx::TensorProto::INT16;
    else if (tensor.getType() == &typeid(int32_t))
        return onnx::TensorProto::INT32;
    else {
        throw std::runtime_error("ONNX_DeepNetExport::getElemType(): "
                                 "tensor type not supported by ONNX!");
    }
}

void N2D2::ONNX_DeepNetExport::addBranchesCells(DeepNet& deepNet) {
    // Need a copy of layers as we will modify the deepNet during the iteration.
    const std::vector<std::vector<std::string>> layers = deepNet.getLayers();

    for(auto itLayer = layers.begin() + 1; itLayer != layers.end(); itLayer++) {
        for(auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            std::shared_ptr<Cell> cell = deepNet.getCell(*itCell);
            if(!cell) {
                throw std::runtime_error("Invalid cell.");
            }

            auto parentsCells = cell->getParentsCells();
            if(parentsCells.size() > 1) {
                if(std::string(cell->getType()) != ElemWiseCell::Type) {
                    auto reg = Registrar<ONNX_ConcatCell>::create(getCellModelType(*cell));
                    auto concatCell = reg(deepNet, 
                                          deepNet.generateNewCellName(cell->getName() + "_concat"), 
                                          cell->getNbChannels());

                    deepNet.addCellBefore(concatCell, cell);
                }
            }
        }
    }
}

std::string N2D2::ONNX_DeepNetExport::getCellModelType(const Cell& cell) {
    const Cell_Frame_Top& cellFrameTop
        = dynamic_cast<const Cell_Frame_Top&>(cell);

    return (cellFrameTop.isCuda())
        ? Cell_Frame_Top::FRAME_CUDA_TYPE
        : Cell_Frame_Top::FRAME_TYPE;
}

#endif
