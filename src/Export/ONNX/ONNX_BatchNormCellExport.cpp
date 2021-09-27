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

#include "DeepNet.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/BatchNormCell.hpp"
#include "Export/BatchNormCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/ONNX/ONNX_CellExport.hpp"
#include "Export/ONNX/ONNX_BatchNormCellExport.hpp"
#include "Export/ONNX/ONNX_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#include <fstream>
#include <string>

N2D2::Registrar<N2D2::ONNX_CellExport>
N2D2::ONNX_BatchNormCellExport::mRegistrarType(
        N2D2::BatchNormCell::Type, N2D2::ONNX_BatchNormCellExport::getInstance);

std::unique_ptr<N2D2::ONNX_BatchNormCellExport>
N2D2::ONNX_BatchNormCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<ONNX_BatchNormCellExport>(new ONNX_BatchNormCellExport);
}

void N2D2::ONNX_BatchNormCellExport::generateNode(
    onnx::GraphProto* graph,
    const DeepNet& deepNet,
    const Cell& cell)
{
    onnx::NodeProto *node = graph->add_node();
    node->set_op_type("BatchNormalization");
    node->set_name(cell.getName());

    // Set parent nodes
    std::vector<std::shared_ptr<Cell> > parents
        = deepNet.getParentCells(cell.getName());

    for (std::vector<std::shared_ptr<Cell> >::const_iterator
        itParent = parents.begin(), itParentEnd = parents.end();
        itParent != itParentEnd; ++itParent)
    {
        node->add_input((*itParent) ? (*itParent)->getName() : "env");
    }

    node->add_input(cell.getName() + "_scale");
    node->add_input(cell.getName() + "_bias");
    node->add_input(cell.getName() + "_mean");
    node->add_input(cell.getName() + "_variance");

    // Set output node
    if (generateActivation(graph, cell))
        node->add_output(cell.getName() + "_act");
    else
        node->add_output(cell.getName());

    // Attributes
    // **********
    const BatchNormCell& bnCell = dynamic_cast<const BatchNormCell&>(cell);

    // Attr epsilon
    onnx::AttributeProto *epsilon_attr = node->add_attribute();
    epsilon_attr->set_name("epsilon");
    epsilon_attr->set_type(onnx::AttributeProto::FLOAT);
    epsilon_attr->set_f(bnCell.getParameter<double>("Epsilon"));

    // Attr momentum
    onnx::AttributeProto *momentum_attr = node->add_attribute();
    momentum_attr->set_name("momentum");
    momentum_attr->set_type(onnx::AttributeProto::FLOAT);
    momentum_attr->set_f(bnCell.getParameter<double>("MovingAverageMomentum"));

    // Scale
    onnx::TensorProto *bn_scale = graph->add_initializer();
    bn_scale->set_name(cell.getName() + "_scale");

    const std::shared_ptr<BaseTensor> scale = bnCell.getScales();
    ONNX_packTensor(bn_scale, *scale, {scale->size()});

    // Bias
    onnx::TensorProto *bn_bias = graph->add_initializer();
    bn_bias->set_name(cell.getName() + "_bias");

    const std::shared_ptr<BaseTensor> bias = bnCell.getBiases();
    ONNX_packTensor(bn_bias, *bias, {bias->size()});

    // Mean
    onnx::TensorProto *bn_mean = graph->add_initializer();
    bn_mean->set_name(cell.getName() + "_mean");

    const std::shared_ptr<BaseTensor> mean = bnCell.getMeans();
    ONNX_packTensor(bn_mean, *mean, {mean->size()});

    // Variance
    onnx::TensorProto *bn_variance = graph->add_initializer();
    bn_variance->set_name(cell.getName() + "_variance");

    const std::shared_ptr<BaseTensor> variance = bnCell.getVariances();
    ONNX_packTensor(bn_variance, *variance, {variance->size()});
}

#endif
