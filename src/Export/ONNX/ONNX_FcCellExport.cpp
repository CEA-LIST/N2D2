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
#include "Cell/FcCell.hpp"
#include "Export/FcCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/ONNX/ONNX_CellExport.hpp"
#include "Export/ONNX/ONNX_FcCellExport.hpp"
#include "Export/ONNX/ONNX_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"
#include "controler/Interface.hpp"

#include <fstream>
#include <string>

N2D2::Registrar<N2D2::ONNX_CellExport>
N2D2::ONNX_FcCellExport::mRegistrarType(
        N2D2::FcCell::Type, N2D2::ONNX_FcCellExport::getInstance);

std::unique_ptr<N2D2::ONNX_FcCellExport>
N2D2::ONNX_FcCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<ONNX_FcCellExport>(new ONNX_FcCellExport);
}

void N2D2::ONNX_FcCellExport::generateNode(
    onnx::GraphProto* graph,
    const DeepNet& deepNet,
    const Cell& cell)
{
    onnx::NodeProto *node = graph->add_node();

    const bool fcInteger = (!mFakeQuantization &&
        CellExport::mPrecision > 0 && CellExport::mPrecision <= 8);

    if (fcInteger)
        node->set_op_type("MatMulInteger");
    else
        node->set_op_type("Gemm");

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

    if (!cell.getParameter<bool>("NoBias") && fcInteger) {
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
    const FcCell& fcCell = dynamic_cast<const FcCell&>(cell);

    if (!fcInteger) {
        // Attr alpha
        onnx::AttributeProto *gemm_alpha = node->add_attribute();
        gemm_alpha->set_name("alpha");
        gemm_alpha->set_type(onnx::AttributeProto::FLOAT);
        gemm_alpha->set_f(1);

        // Attr beta
        onnx::AttributeProto *gemm_beta = node->add_attribute();
        gemm_beta->set_name("beta");
        gemm_beta->set_type(onnx::AttributeProto::FLOAT);
        gemm_beta->set_f(!cell.getParameter<bool>("NoBias"));

        // Attr transA
        onnx::AttributeProto *gemm_transA = node->add_attribute();
        gemm_transA->set_name("transA");
        gemm_transA->set_type(onnx::AttributeProto::INT);
        gemm_transA->set_i(0);

        // Attr transB
        onnx::AttributeProto *gemm_transB = node->add_attribute();
        gemm_transB->set_name("transB");
        gemm_transB->set_type(onnx::AttributeProto::INT);
        gemm_transB->set_i(1);
    }

    // Weights input
    onnx::TensorProto *fc_w = graph->add_initializer();
    fc_w->set_name(cell.getName() + "_w");

    const BaseInterface* weightsInterface = fcCell.getWeights();
    assert(weightsInterface->size() == 1);
    const BaseTensor& weights = (*weightsInterface)[0U];

    if (fcInteger) {
        const Tensor<Float_T>& weightsFloat = tensor_cast<Float_T>(weights);
        Tensor<Float_T> weightsT({weights.dimB(),
                                  weights.size() / weights.dimB()});

        for (int i = 0; i < weights.size() / weights.dimB(); ++i) {
            for (int n = 0; n < weights.dimB(); ++n)
                weightsT(n, i) = weightsFloat(i, n);
        }

        ONNX_castAndPackTensor(mPrecision, fc_w, weightsT,
            {fcCell.getInputsSize(), fcCell.getNbOutputs()});
    }
    else {
        ONNX_castAndPackTensor(mPrecision, fc_w, weights,
            {fcCell.getInputsSize(), fcCell.getNbOutputs()});
    }

    // Bias input
    if (!cell.getParameter<bool>("NoBias")) {
        if (fcInteger) {
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

        onnx::TensorProto *fc_b = graph->add_initializer();
        fc_b->set_name(cell.getName() + "_b");

        const BaseTensor* biases = fcCell.getBiases();
        ONNX_castAndPackTensor((mPrecision > 0) ? 4 * mPrecision : mPrecision,
            fc_b, *biases, {biases->size()});
    }
}

#endif
