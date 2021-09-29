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
#include "Cell/SoftmaxCell.hpp"
#include "Export/SoftmaxCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/ONNX/ONNX_CellExport.hpp"
#include "Export/ONNX/ONNX_SoftmaxCellExport.hpp"
#include "Export/ONNX/ONNX_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#include <fstream>
#include <string>

N2D2::Registrar<N2D2::ONNX_CellExport>
N2D2::ONNX_SoftmaxCellExport::mRegistrarType(
        N2D2::SoftmaxCell::Type, N2D2::ONNX_SoftmaxCellExport::getInstance);

std::unique_ptr<N2D2::ONNX_SoftmaxCellExport>
N2D2::ONNX_SoftmaxCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<ONNX_SoftmaxCellExport>(new ONNX_SoftmaxCellExport);
}

void N2D2::ONNX_SoftmaxCellExport::generateNode(
    onnx::GraphProto* graph,
    const DeepNet& deepNet,
    const Cell& cell)
{
    onnx::NodeProto *node = graph->add_node();
    node->set_op_type("Softmax");
    node->set_name(cell.getName());

    // Set parent nodes
    std::vector<std::shared_ptr<Cell> > parents
        = deepNet.getParentCells(cell.getName());

    for (std::vector<std::shared_ptr<Cell> >::const_iterator
        itParent = parents.begin(), itParentEnd = parents.end();
        itParent != itParentEnd; ++itParent)
    {
        const std::string parentName = (*itParent)
            ? (*itParent)->getName() : "env";

        if (CellExport::mPrecision > 0) {
            // Softmax is never quantized. If it is kept in the export, the
            // input must be casted to float.
            node->add_input(ONNX_castInput(graph, parentName,
                onnx::TensorProto::FLOAT));
        }
        else
            node->add_input(parentName);
    }

    // Set output node to the name of the cell
    node->add_output(cell.getName());

    // Attributes
    // **********
    // Attr axis
    onnx::AttributeProto *softmax_axis = node->add_attribute();
    softmax_axis->set_name("axis");
    softmax_axis->set_type(onnx::AttributeProto::INT);
    softmax_axis->set_i(1);
}

#endif
