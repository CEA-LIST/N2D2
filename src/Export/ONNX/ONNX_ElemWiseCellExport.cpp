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
#include "Cell/ElemWiseCell.hpp"
#include "Export/ElemWiseCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/ONNX/ONNX_CellExport.hpp"
#include "Export/ONNX/ONNX_ElemWiseCellExport.hpp"
#include "Export/ONNX/ONNX_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#include <fstream>
#include <string>

N2D2::Registrar<N2D2::ONNX_CellExport>
N2D2::ONNX_ElemWiseCellExport::mRegistrarType(
        N2D2::ElemWiseCell::Type, N2D2::ONNX_ElemWiseCellExport::getInstance);

std::unique_ptr<N2D2::ONNX_ElemWiseCellExport>
N2D2::ONNX_ElemWiseCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<ONNX_ElemWiseCellExport>(new ONNX_ElemWiseCellExport);
}

void N2D2::ONNX_ElemWiseCellExport::generateNode(
    onnx::GraphProto* graph,
    const DeepNet& deepNet,
    const Cell& cell)
{
    const ElemWiseCell& elemWiseCell = dynamic_cast<const ElemWiseCell&>(cell);
    const ElemWiseCell::Operation elemOp = elemWiseCell.getOperation();
    std::vector<std::shared_ptr<Cell> > parents
        = deepNet.getParentCells(cell.getName());

    onnx::NodeProto *node = graph->add_node();

    if (elemOp == ElemWiseCell::Sum || elemOp == ElemWiseCell::AbsSum)
        node->set_op_type("Sum");
    else if (elemOp == ElemWiseCell::Prod && parents.size() == 2)
        node->set_op_type("Mul");
    else if (elemOp == ElemWiseCell::Max)
        node->set_op_type("Max");
    else {
        std::ostringstream msgStr;
        msgStr << "ONNX_ElemWiseCellExport::generateNode(): operation "
            << elemOp << " not supported in ONNX.";

        throw std::runtime_error(msgStr.str());
    }

    if (elemWiseCell.getCoeffMode() == ElemWiseCell::PerChannel) {
        std::ostringstream msgStr;
        msgStr << "ONNX_ElemWiseCellExport::generateNode(): mode "
            << elemWiseCell.getCoeffMode() << " not supported in ONNX.";

        throw std::runtime_error(msgStr.str());
    }

    node->set_name(cell.getName());

    // Set parent nodes
    for (std::vector<std::shared_ptr<Cell> >::const_iterator
        itParent = parents.begin(), itParentEnd = parents.end();
        itParent != itParentEnd; ++itParent)
    {
        const std::string parentName = (*itParent)
            ? (*itParent)->getName() : "env";

        if (elemOp == ElemWiseCell::AbsSum) {
            onnx::NodeProto *node_abs = graph->add_node();
            node_abs->set_op_type("Abs");
            node_abs->set_name(parentName + "_abs");
            node_abs->add_input(parentName);
            node_abs->add_output(parentName + "_abs");
            node->add_input(parentName + "_abs");
        }
        else
            node->add_input(parentName);
    }

    // Set output node
    if (generateActivation(graph, cell))
        node->add_output(cell.getName() + "_act");
    else
        node->add_output(cell.getName());
}

#endif
