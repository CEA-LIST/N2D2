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

#include "Activation/LinearActivation.hpp"
#include "Activation/LogisticActivation.hpp"
#include "Activation/RectifierActivation.hpp"
#include "Activation/SaturationActivation.hpp"
#include "Activation/SoftplusActivation.hpp"
#include "Activation/TanhActivation.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/CellExport.hpp"
#include "Export/ONNX/ONNX_CellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "utils/Utils.hpp"

void N2D2::ONNX_CellExport::ONNX_castAndPackTensor(
    onnx::TensorProto* onnxTensor,
    const BaseTensor& tensor,
    const std::vector<size_t>& shape)
{
    if (CellExport::mPrecision > 0) {
        if (CellExport::mPrecision <= 8) {
            const Tensor<int8_t> tensor_int
                = tensor_cast<int8_t, true>(tensor);
            ONNX_packTensor(onnxTensor, tensor_int, shape);
        }
        else if (CellExport::mPrecision <= 16) {
            const Tensor<int16_t> tensor_int
                = tensor_cast<int16_t, true>(tensor);
            ONNX_packTensor(onnxTensor, tensor_int, shape);
        }
        else {
            const Tensor<int32_t> tensor_int
                = tensor_cast<int32_t, true>(tensor);
            ONNX_packTensor(onnxTensor, tensor_int, shape);
        }
    }
    else
        ONNX_packTensor(onnxTensor, tensor, shape);
}

void N2D2::ONNX_CellExport::ONNX_packTensor(
    onnx::TensorProto* onnxTensor,
    const BaseTensor& tensor,
    const std::vector<size_t>& shape)
{
    if (tensor.getType() == &typeid(float)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<float>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(half_float::half)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<half_float::half>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(double)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<double>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(int8_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<int8_t>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(int16_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<int16_t>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(int32_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<int32_t>&>(tensor), shape);
    }
    else {
        throw std::runtime_error("ONNX_CellExport::ONNX_packTensor(): "
                                 "tensor type not supported by ONNX!");
    }
}

namespace N2D2 {
template <>
void ONNX_CellExport::ONNX_packTensor<float>(
    onnx::TensorProto* onnxTensor,
    const Tensor<float>& tensor,
    const std::vector<size_t>& shape)
{
    onnxTensor->set_data_type(onnx::TensorProto::FLOAT);

    std::vector<size_t> dims = (!shape.empty()) ? shape : tensor.dims();
    std::reverse(dims.begin(), dims.end());
    std::for_each(dims.begin(), dims.end(), [&onnxTensor](size_t dim)
        { onnxTensor->mutable_dims()->Add(dim); });
    onnxTensor->set_raw_data(&tensor.data().data()[0],
        sizeof(float) * tensor.size());
}

template <>
void ONNX_CellExport::ONNX_packTensor<half_float::half>(
    onnx::TensorProto* onnxTensor,
    const Tensor<half_float::half>& tensor,
    const std::vector<size_t>& shape)
{
    onnxTensor->set_data_type(onnx::TensorProto::FLOAT16);

    std::vector<size_t> dims = (!shape.empty()) ? shape : tensor.dims();
    std::reverse(dims.begin(), dims.end());
    std::for_each(dims.begin(), dims.end(), [&onnxTensor](size_t dim)
        { onnxTensor->mutable_dims()->Add(dim); });
    onnxTensor->set_raw_data(&tensor.data().data()[0],
        sizeof(half_float::half) * tensor.size());
}

template <>
void ONNX_CellExport::ONNX_packTensor<double>(
    onnx::TensorProto* onnxTensor,
    const Tensor<double>& tensor,
    const std::vector<size_t>& shape)
{
    onnxTensor->set_data_type(onnx::TensorProto::DOUBLE);

    std::vector<size_t> dims = (!shape.empty()) ? shape : tensor.dims();
    std::reverse(dims.begin(), dims.end());
    std::for_each(dims.begin(), dims.end(), [&onnxTensor](size_t dim)
        { onnxTensor->mutable_dims()->Add(dim); });
    onnxTensor->set_raw_data(&tensor.data().data()[0],
        sizeof(double) * tensor.size());
}

template <>
void ONNX_CellExport::ONNX_packTensor<int8_t>(
    onnx::TensorProto* onnxTensor,
    const Tensor<int8_t>& tensor,
    const std::vector<size_t>& shape)
{
    onnxTensor->set_data_type(onnx::TensorProto::INT8);

    std::vector<size_t> dims = (!shape.empty()) ? shape : tensor.dims();
    std::reverse(dims.begin(), dims.end());
    std::for_each(dims.begin(), dims.end(), [&onnxTensor](size_t dim)
        { onnxTensor->mutable_dims()->Add(dim); });
    onnxTensor->set_raw_data(&tensor.data().data()[0],
        sizeof(int8_t) * tensor.size());
}

template <>
void ONNX_CellExport::ONNX_packTensor<int16_t>(
    onnx::TensorProto* onnxTensor,
    const Tensor<int16_t>& tensor,
    const std::vector<size_t>& shape)
{
    onnxTensor->set_data_type(onnx::TensorProto::INT16);

    std::vector<size_t> dims = (!shape.empty()) ? shape : tensor.dims();
    std::reverse(dims.begin(), dims.end());
    std::for_each(dims.begin(), dims.end(), [&onnxTensor](size_t dim)
        { onnxTensor->mutable_dims()->Add(dim); });
    onnxTensor->set_raw_data(&tensor.data().data()[0],
        sizeof(int16_t) * tensor.size());
}

template <>
void ONNX_CellExport::ONNX_packTensor<int32_t>(
    onnx::TensorProto* onnxTensor,
    const Tensor<int32_t>& tensor,
    const std::vector<size_t>& shape)
{
    onnxTensor->set_data_type(onnx::TensorProto::INT32);

    std::vector<size_t> dims = (!shape.empty()) ? shape : tensor.dims();
    std::reverse(dims.begin(), dims.end());
    std::for_each(dims.begin(), dims.end(), [&onnxTensor](size_t dim)
        { onnxTensor->mutable_dims()->Add(dim); });
    onnxTensor->set_raw_data(&tensor.data().data()[0],
        sizeof(int32_t) * tensor.size());
}
}

bool N2D2::ONNX_CellExport::generateActivation(onnx::GraphProto* graph,
                                               const Cell& cell)
{
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    const std::shared_ptr<Activation>& activation = cellFrame.getActivation();

    if (!activation || activation->getType() == LinearActivation::Type)
        return false;

    onnx::NodeProto *node = graph->add_node();
    node->set_name(cell.getName() + "_act");
    node->add_input(cell.getName() + "_act");
    node->add_output(cell.getName());

    if (activation->getType() == LogisticActivation::Type)
        node->set_op_type("Sigmoid");
    else if (activation->getType() == RectifierActivation::Type) {
        const double leakSlope = activation->getParameter<double>("LeakSlope");
        const double clipping = activation->getParameter<double>("Clipping");

        if (leakSlope > 0.0 && clipping == 0.0) {
            node->set_op_type("LeakyRelu");

            // Attr alpha
            onnx::AttributeProto *alpha_attr = node->add_attribute();
            alpha_attr->set_name("alpha");
            alpha_attr->set_type(onnx::AttributeProto::FLOAT);
            alpha_attr->set_f(leakSlope);
        }
        else if (leakSlope == 0.0 && clipping > 0.0) {
            node->set_op_type("Clip");

            // Attr min
            onnx::AttributeProto *min_attr = node->add_attribute();
            min_attr->set_name("min");
            min_attr->set_type(onnx::AttributeProto::FLOAT);
            min_attr->set_f(0.0);

            // Attr max
            onnx::AttributeProto *max_attr = node->add_attribute();
            max_attr->set_name("max");
            max_attr->set_type(onnx::AttributeProto::FLOAT);
            max_attr->set_f(clipping);
        }
        else
            node->set_op_type("Relu");
    }
    else if (activation->getType() == SaturationActivation::Type) {
        const double threshold = activation->getParameter<double>("Threshold");

        node->set_op_type("Clip");

        // Attr min
        onnx::AttributeProto *min_attr = node->add_attribute();
        min_attr->set_name("min");
        min_attr->set_type(onnx::AttributeProto::FLOAT);
        min_attr->set_f(-threshold);

        // Attr max
        onnx::AttributeProto *max_attr = node->add_attribute();
        max_attr->set_name("max");
        max_attr->set_type(onnx::AttributeProto::FLOAT);
        max_attr->set_f(threshold);
    }
    else if (activation->getType() == SoftplusActivation::Type)
        node->set_op_type("Softplus");
    else if (activation->getType() == TanhActivation::Type)
        node->set_op_type("Tanh");
    else {
        std::ostringstream msgStr;
        msgStr << "ONNX_CellExport::generateActivation(): activation "
            << activation->getType() << " not supported in ONNX.";

        throw std::runtime_error(msgStr.str());
    }

    return true;
}

#endif
