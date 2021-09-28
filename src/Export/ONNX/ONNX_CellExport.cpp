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

bool N2D2::ONNX_CellExport::mImplicitCasting = false;
bool N2D2::ONNX_CellExport::mFakeQuantization = false;

bool N2D2::ONNX_CellExport::generateActivation(onnx::GraphProto* graph,
                                               const Cell& cell)
{
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    const std::shared_ptr<Activation>& activation = cellFrame.getActivation();
    const Scaling& scaling = activation->getActivationScaling();

    if (!activation || (activation->getType() == LinearActivation::Type
        && scaling.getMode() == ScalingMode::NONE))
    {
        return false;
    }

    std::string inputName = cell.getName() + "_act";

    if (activation->getType() != LinearActivation::Type) {
        // Activation
        // Activation remains in the same data type as the layer
        onnx::NodeProto *node = graph->add_node();
        node->set_name(inputName);
        node->add_input(inputName);

        if (scaling.getMode() != ScalingMode::NONE)
            node->add_output(inputName + "_scaling");
        else
            node->add_output(cell.getName());

        if (activation->getType() == LogisticActivation::Type)
            node->set_op_type("Sigmoid");
        else if (activation->getType() == RectifierActivation::Type) {
            const double leakSlope
                = activation->getParameter<double>("LeakSlope");
            const double clipping
                = activation->getParameter<double>("Clipping");

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
            const double threshold
                = activation->getParameter<double>("Threshold");

            node->set_op_type("Clip");
            node->add_input(inputName + "_min");
            node->add_input(inputName + "_max");

            onnx::TensorProto *clip_min = graph->add_initializer();
            clip_min->set_name(inputName + "_min");
            ONNX_packTensor(clip_min, Tensor<float>({1}, -threshold));

            onnx::TensorProto *clip_max = graph->add_initializer();
            clip_max->set_name(inputName + "_max");
            ONNX_packTensor(clip_max, Tensor<float>({1}, threshold));
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

        inputName += "_scaling";
    }

    if (scaling.getMode() != ScalingMode::NONE) {
        // Scaling
        // If there is scaling, the data type is assumed to be INT32,
        // in accordance with the ConvInteger and MatMulInteger ONNX operator
        // outputs.
        onnx::NodeProto *node = graph->add_node();
        node->set_name(inputName);

        if (scaling.getMode() == ScalingMode::FLOAT_MULT) {
            // For FLOAT_MULT, the input must first be casted to FLOAT
            node->add_input(ONNX_castInput(graph, inputName,
                                           onnx::TensorProto::FLOAT));

            node->set_op_type("Mul");
            node->add_input(inputName + "_w");

            onnx::TensorProto *scaling_w = graph->add_initializer();
            scaling_w->set_name(inputName + "_w");

            const auto& scalingPerOutput = scaling.getFloatingPointScaling()
                                                        .getScalingPerOutput();
            
            if (Utils::all_same(scalingPerOutput.begin(),
                                scalingPerOutput.end()))
            {
                const Tensor<float> w({1}, scalingPerOutput.front());
                ONNX_packTensor(scaling_w, w);
            }
            else {
                const Tensor<float> w({scalingPerOutput.size()},
                    scalingPerOutput.begin(),
                    scalingPerOutput.end());
                ONNX_packTensor(scaling_w, w);
            }
        }
        else if (scaling.getMode() == ScalingMode::SINGLE_SHIFT) {
            // For SINGLE_SHIFT, the input must first be casted to UNSIGNED,
            // as required by the BitShift ONNX operator.
            node->add_input(ONNX_castInput(graph, inputName,
                (CellExport::mPrecision <= 8) ? onnx::TensorProto::UINT32
                                              : onnx::TensorProto::UINT64));
            node->add_input(inputName + "_s");

            onnx::TensorProto *scaling_w = graph->add_initializer();
            scaling_w->set_name(inputName + "_s");

            const auto& scalingPerOutput = scaling.getSingleShiftScaling()
                                                        .getScalingPerOutput();

            if (mFakeQuantization) {
                node->set_op_type("Mul");

                std::vector<float> mulScaling;
                std::transform(scalingPerOutput.begin(), scalingPerOutput.end(),
                    std::back_inserter(mulScaling), [](unsigned char shift)
                        { return 1.0 / (1ll << shift); });

                if (Utils::all_same(scalingPerOutput.begin(),
                                    scalingPerOutput.end()))
                {
                    const Tensor<float> w({1}, mulScaling.front());
                    ONNX_packTensor(scaling_w, w);
                }
                else {
                    const Tensor<float> w({mulScaling.size()},
                        mulScaling.begin(),
                        mulScaling.end());
                    ONNX_packTensor(scaling_w, w);
                }
            }
            else {
                node->set_op_type("BitShift");

                // Attr direction
                onnx::AttributeProto *shift_direction = node->add_attribute();
                shift_direction->set_name("direction");
                shift_direction->set_type(onnx::AttributeProto::STRING);
                shift_direction->set_s("RIGHT");

                if (Utils::all_same(scalingPerOutput.begin(),
                                    scalingPerOutput.end()))
                {
                    const Tensor<uint32_t> w({1}, scalingPerOutput.front());
                    ONNX_packTensor(scaling_w, w);
                }
                else {
                    const Tensor<uint32_t> w({scalingPerOutput.size()},
                        scalingPerOutput.begin(),
                        scalingPerOutput.end());
                    ONNX_packTensor(scaling_w, w);
                }
            }
        }
        else {
            std::ostringstream msgStr;
            msgStr << "ONNX_CellExport::generateActivation(): scaling mode"
                " not supported in ONNX.";

            throw std::runtime_error(msgStr.str());
        }

        // Output of scaling should be of the same type as the output of the
        // activation
        node->add_output(ONNX_castOutput(graph, inputName + "_clip",
            (CellExport::mPrecision <= 8) ? onnx::TensorProto::INT32
                                          : onnx::TensorProto::INT64));
        inputName += "_clip";

        // Clipping
        onnx::NodeProto *nodeClip = graph->add_node();
        nodeClip->set_name(inputName);
        nodeClip->set_op_type("Clip");
        nodeClip->add_input(inputName);
        nodeClip->add_input(inputName + "_min");
        nodeClip->add_input(inputName + "_max");

        const bool isOutputUnsigned = DeepNetExport::isCellOutputUnsigned(cell);
        const int32_t minVal = (isOutputUnsigned)
            ? 0
            : -(1ll << (CellExport::mPrecision - 1ll));
        const int32_t maxVal = (isOutputUnsigned)
            ? (1ll << CellExport::mPrecision) - 1ll
            : (1ll << (CellExport::mPrecision - 1ll)) - 1ll;

        onnx::TensorProto *clip_min = graph->add_initializer();
        clip_min->set_name(inputName + "_min");
        onnx::TensorProto *clip_max = graph->add_initializer();
        clip_max->set_name(inputName + "_max");

        ONNX_castAndPackTensor((CellExport::mPrecision <= 8) ? 32 : 64,
                               clip_min, Tensor<int32_t>({1}, minVal));
        ONNX_castAndPackTensor((CellExport::mPrecision <= 8) ? 32 : 64,
                               clip_max, Tensor<int32_t>({1}, maxVal));

        const onnx::TensorProto::DataType outputType
            = (CellExport::mPrecision <= 8)
                ? ((isOutputUnsigned) ? onnx::TensorProto::UINT8
                                      : onnx::TensorProto::INT8) :
              (CellExport::mPrecision <= 16)
                ? ((isOutputUnsigned) ? onnx::TensorProto::UINT16
                                      : onnx::TensorProto::INT16) :
                  ((isOutputUnsigned) ? onnx::TensorProto::UINT32
                                      : onnx::TensorProto::INT32);

        nodeClip->add_output(ONNX_castOutput(graph, cell.getName(),
            outputType));
    }

    return true;
}

std::string N2D2::ONNX_CellExport::ONNX_castInput(
    onnx::GraphProto* graph,
    const std::string& input,
    onnx::TensorProto::DataType to)
{
    if (mImplicitCasting || mFakeQuantization)
        return input;

    onnx::NodeProto *node = graph->add_node();
    node->set_op_type("Cast");
    node->set_name(input + "_icast");
    node->add_input(input);
    node->add_output(input + "_icast");

    // Attr to
    onnx::AttributeProto *cast_to = node->add_attribute();
    cast_to->set_name("to");
    cast_to->set_type(onnx::AttributeProto::INT);
    cast_to->set_i(to);

    return (input + "_icast");
}

std::string N2D2::ONNX_CellExport::ONNX_castOutput(
    onnx::GraphProto* graph,
    const std::string& output,
    onnx::TensorProto::DataType to)
{
    if (mImplicitCasting || mFakeQuantization)
        return output;

    onnx::NodeProto *node = graph->add_node();
    node->set_op_type("Cast");
    node->set_name(output + "_ocast");
    node->add_input(output + "_ocast");
    node->add_output(output);

    // Attr to
    onnx::AttributeProto *cast_to = node->add_attribute();
    cast_to->set_name("to");
    cast_to->set_type(onnx::AttributeProto::INT);
    cast_to->set_i(to);

    return (output + "_ocast");
}

void N2D2::ONNX_CellExport::ONNX_castAndPackTensor(
    int precision,
    onnx::TensorProto* onnxTensor,
    const BaseTensor& tensor,
    const std::vector<size_t>& shape)
{
    if (mFakeQuantization)
        precision = -32;

    if (precision > 0 && precision <= 8) {
        const Tensor<int8_t> tensor_int
            = tensor_cast<int8_t, true>(tensor);
        ONNX_packTensor(onnxTensor, tensor_int, shape);
    }
    else if (precision > 8 && precision <= 16) {
        const Tensor<int16_t> tensor_int
            = tensor_cast<int16_t, true>(tensor);
        ONNX_packTensor(onnxTensor, tensor_int, shape);
    }
    else if (precision > 16) {
        const Tensor<int32_t> tensor_int
            = tensor_cast<int32_t, true>(tensor);
        ONNX_packTensor(onnxTensor, tensor_int, shape);
    }
    else if (precision == -16) {
        const Tensor<half_float::half> tensor_float
            = tensor_cast<half_float::half>(tensor);
        ONNX_packTensor(onnxTensor, tensor_float, shape);
    }
    else if (precision == -32) {
        const Tensor<float> tensor_float
            = tensor_cast<float>(tensor);
        ONNX_packTensor(onnxTensor, tensor_float, shape);
    }
    else if (precision == -64) {
        const Tensor<double> tensor_float
            = tensor_cast<double>(tensor);
        ONNX_packTensor(onnxTensor, tensor_float, shape);
    }
    else {
        throw std::runtime_error("ONNX_CellExport::ONNX_castAndPackTensor(): "
                                 "unknown precision!");
    }
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
    else if (tensor.getType() == &typeid(uint8_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<uint8_t>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(int16_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<int16_t>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(uint16_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<uint16_t>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(int32_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<int32_t>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(uint32_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<uint32_t>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(int64_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<int64_t>&>(tensor), shape);
    }
    else if (tensor.getType() == &typeid(uint64_t)) {
        return ONNX_packTensor(onnxTensor,
            dynamic_cast<const Tensor<uint64_t>&>(tensor), shape);
    }
    else {
        throw std::runtime_error("ONNX_CellExport::ONNX_packTensor(): "
                                 "tensor type not supported by ONNX!");
    }
}

namespace N2D2 {
template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<float>()
{
    return onnx::TensorProto::FLOAT;
}

template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<half_float::half>()
{
    return onnx::TensorProto::FLOAT16;
}

template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<double>()
{
    return onnx::TensorProto::DOUBLE;
}

template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<int8_t>()
{
    return onnx::TensorProto::INT8;
}

template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<uint8_t>()
{
    return onnx::TensorProto::UINT8;
}

template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<int16_t>()
{
    return onnx::TensorProto::INT16;
}

template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<uint16_t>()
{
    return onnx::TensorProto::UINT16;
}

template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<int32_t>()
{
    return onnx::TensorProto::INT32;
}

template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<uint32_t>()
{
    return onnx::TensorProto::UINT32;
}

template <>
onnx::TensorProto::DataType ONNX_CellExport::ONNX_dataType<int64_t>()
{
    return onnx::TensorProto::INT64;
}

template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<uint64_t>()
{
    return onnx::TensorProto::UINT64;
}
}

#endif
