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
#include "Cell/ConvCell.hpp"
#include "Export/ConvCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/ONNX/ONNX_CellExport.hpp"
#include "Export/ONNX/ONNX_ConvCellExport.hpp"
#include "Export/ONNX/ONNX_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#include <fstream>
#include <string>

N2D2::Registrar<N2D2::ONNX_CellExport>
N2D2::ONNX_ConvCellExport::mRegistrarType(
        N2D2::ConvCell::Type, N2D2::ONNX_ConvCellExport::getInstance);

std::unique_ptr<N2D2::ONNX_ConvCellExport>
N2D2::ONNX_ConvCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<ONNX_ConvCellExport>(new ONNX_ConvCellExport);
}

void N2D2::ONNX_ConvCellExport::generateNode(
    onnx::GraphProto* graph,
    const DeepNet& deepNet,
    const Cell& cell)
{
    onnx::NodeProto *node = graph->add_node();

    const bool convInteger = (!mFakeQuantization &&
        CellExport::mPrecision > 0 && CellExport::mPrecision <= 8);

    if (convInteger)
        node->set_op_type("ConvInteger");
    else
        node->set_op_type("Conv");

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

    // Set weights input
    node->add_input(cell.getName() + "_w");

    if (!cell.getParameter<bool>("NoBias") && convInteger) {
        node->add_output(cell.getName() + "_bias");
    }
    else {
        if (!cell.getParameter<bool>("NoBias"))
            node->add_input(cell.getName() + "_b");

        if (generateActivation(graph, cell))
            node->add_output(cell.getName() + "_act");
        else
            node->add_output(cell.getName());
    }

    // Attributes
    // **********
    const ConvCell& convCell = dynamic_cast<const ConvCell&>(cell);

    // Attr dilations
    onnx::AttributeProto *conv_dilations = node->add_attribute();
    conv_dilations->set_name("dilations");
    conv_dilations->set_type(onnx::AttributeProto::INTS);

    std::vector<unsigned int> dilationDims = convCell.getDilationDims();
    std::reverse(dilationDims.begin(), dilationDims.end());

    for (int dim : dilationDims)
        conv_dilations->add_ints(dim);

    // Attr group
    onnx::AttributeProto *conv_group = node->add_attribute();
    conv_group->set_name("group");
    conv_group->set_type(onnx::AttributeProto::INT);
    conv_group->set_i((convCell.groupMap() > 1) ? convCell.groupMap() : 1);

    // Attr kernel_shape
    onnx::AttributeProto *conv_kernel_shape = node->add_attribute();
    conv_kernel_shape->set_name("kernel_shape");
    conv_kernel_shape->set_type(onnx::AttributeProto::INTS);

    std::vector<unsigned int> kernelDims = convCell.getKernelDims();
    std::reverse(kernelDims.begin(), kernelDims.end());

    for (int dim : kernelDims)
        conv_kernel_shape->add_ints(dim);

    // Attr pads
    onnx::AttributeProto *conv_pads = node->add_attribute();
    conv_pads->set_name("pads");
    conv_pads->set_type(onnx::AttributeProto::INTS);

    std::vector<int> paddingDims = convCell.getExtendedPadding();
    paddingDims[0] += convCell.getPaddingX();  // X_L
    paddingDims[1] += convCell.getPaddingY();  // Y_T
    paddingDims[2] += convCell.getPaddingX();  // X_R
    paddingDims[3] += convCell.getPaddingY();  // Y_B

    const size_t half = paddingDims.size() / 2;

    for (size_t dim = 0; dim < half; ++dim)
        conv_pads->add_ints(paddingDims[half - 1 - dim]);

    for (size_t dim = 0; dim < half; ++dim)
        conv_pads->add_ints(paddingDims[paddingDims.size() - 1 - dim]);

    // Attr strides
    onnx::AttributeProto *conv_strides = node->add_attribute();
    conv_strides->set_name("strides");
    conv_strides->set_type(onnx::AttributeProto::INTS);

    std::vector<unsigned int> strideDims = convCell.getStrideDims();
    std::reverse(strideDims.begin(), strideDims.end());

    for (int dim : strideDims)
        conv_strides->add_ints(dim);

    // Weights input
    onnx::TensorProto *conv_w = graph->add_initializer();
    conv_w->set_name(cell.getName() + "_w");

    const BaseInterface* weightsInterface = convCell.getWeights();
    const BaseTensor& weights = (*weightsInterface)[0];

    ONNX_castAndPackTensor(mPrecision, conv_w, weights);

    // Bias input
    if (!cell.getParameter<bool>("NoBias")) {
        if (convInteger) {
            onnx::NodeProto *nodeBias = graph->add_node();
            nodeBias->set_op_type("Add");
            nodeBias->set_name(cell.getName() + "_bias");
            nodeBias->add_input(cell.getName() + "_bias");
            nodeBias->add_input(cell.getName() + "_b");
                    
            if (generateActivation(graph, cell))
                nodeBias->add_output(cell.getName() + "_act");
            else
                nodeBias->add_output(cell.getName());
        }

        onnx::TensorProto *conv_b = graph->add_initializer();
        conv_b->set_name(cell.getName() + "_b");

        const std::shared_ptr<BaseTensor> biases = convCell.getBiases();
        ONNX_castAndPackTensor((mPrecision > 0) ? 4 * mPrecision : mPrecision,
            conv_b, *biases, {biases->size()});
    }
}

#endif
