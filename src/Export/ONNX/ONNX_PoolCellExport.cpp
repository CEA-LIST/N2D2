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
#include "Cell/PoolCell.hpp"
#include "Export/PoolCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/ONNX/ONNX_CellExport.hpp"
#include "Export/ONNX/ONNX_PoolCellExport.hpp"
#include "Export/ONNX/ONNX_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#include <fstream>
#include <string>

N2D2::Registrar<N2D2::ONNX_CellExport>
N2D2::ONNX_PoolCellExport::mRegistrarType(
        N2D2::PoolCell::Type, N2D2::ONNX_PoolCellExport::getInstance);

std::unique_ptr<N2D2::ONNX_PoolCellExport>
N2D2::ONNX_PoolCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<ONNX_PoolCellExport>(new ONNX_PoolCellExport);
}

void N2D2::ONNX_PoolCellExport::generateNode(
    onnx::GraphProto* graph,
    const DeepNet& deepNet,
    const Cell& cell)
{
    const PoolCell& poolCell = dynamic_cast<const PoolCell&>(cell);

    onnx::NodeProto *node = graph->add_node();

    if (poolCell.getPooling() == PoolCell::Max)
        node->set_op_type("MaxPool");
    else if (poolCell.getPooling() == PoolCell::Average)
        node->set_op_type("AveragePool");
    else {
        std::ostringstream msgStr;
        msgStr << "ONNX_PoolCellExport::generateNode(): pooling "
            << poolCell.getPooling() << " not supported in ONNX.";

        throw std::runtime_error(msgStr.str());
    }

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

    // Set output node
    if (generateActivation(graph, cell))
        node->add_output(cell.getName() + "_act");
    else
        node->add_output(cell.getName());

    // Attributes
    // **********
    // Attr kernel_shape
    onnx::AttributeProto *pool_kernel_shape = node->add_attribute();
    pool_kernel_shape->set_name("kernel_shape");
    pool_kernel_shape->set_type(onnx::AttributeProto::INTS);

    std::vector<unsigned int> kernelDims = poolCell.getPoolDims();
    std::reverse(kernelDims.begin(), kernelDims.end());

    for (int dim : kernelDims)
        pool_kernel_shape->add_ints(dim);

    // Attr pads
    onnx::AttributeProto *pool_pads = node->add_attribute();
    pool_pads->set_name("pads");
    pool_pads->set_type(onnx::AttributeProto::INTS);

    std::vector<int> paddingDims = poolCell.getExtendedPadding();
    paddingDims[0] += poolCell.getPaddingX();  // X_L
    paddingDims[1] += poolCell.getPaddingY();  // Y_T
    paddingDims[2] += poolCell.getPaddingX();  // X_R
    paddingDims[3] += poolCell.getPaddingY();  // Y_B

    const size_t half = paddingDims.size() / 2;

    for (size_t dim = 0; dim < half; ++dim)
        pool_pads->add_ints(paddingDims[half - 1 - dim]);

    for (size_t dim = 0; dim < half; ++dim)
        pool_pads->add_ints(paddingDims[paddingDims.size() - 1 - dim]);

    // Attr strides
    onnx::AttributeProto *pool_strides = node->add_attribute();
    pool_strides->set_name("strides");
    pool_strides->set_type(onnx::AttributeProto::INTS);

    std::vector<unsigned int> strideDims = poolCell.getStrideDims();
    std::reverse(strideDims.begin(), strideDims.end());

    for (int dim : strideDims)
        pool_strides->add_ints(dim);
}

#endif
