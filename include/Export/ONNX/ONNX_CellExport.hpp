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

#ifndef N2D2_ONNX_CELLEXPORT_H
#define N2D2_ONNX_CELLEXPORT_H

#ifdef ONNX

#include <fstream>
#include <functional>
#include <memory>
#include <string>

#include "Cell/Cell.hpp"
#include "Scaling.hpp"
#include "utils/Registrar.hpp"

#include <onnx.pb.h>

namespace N2D2 {
/**
 * Virtual base class for methods commun to every cell type for the ONNX export
 * ANY CELL, ONNX EXPORT
*/
class ONNX_CellExport {
public:
    typedef std::function
        <std::unique_ptr<ONNX_CellExport>(Cell&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static bool mImplicitCasting;
    static bool mFakeQuantization;

    static std::string ONNX_castInput(
        onnx::GraphProto* graph,
        const std::string& input,
        onnx::TensorProto::DataType to);
    static std::string ONNX_castOutput(
        onnx::GraphProto* graph,
        const std::string& output,
        onnx::TensorProto::DataType to);
    static void ONNX_castAndPackTensor(int precision,
                                       onnx::TensorProto* onnxTensor,
                                       const BaseTensor& tensor,
                                       const std::vector<size_t>& shape
                                                    = std::vector<size_t>());
    static void ONNX_packTensor(onnx::TensorProto* onnxTensor,
                                const BaseTensor& tensor,
                                const std::vector<size_t>& shape
                                                    = std::vector<size_t>());
    template <class T>
    static onnx::TensorProto::DataType ONNX_dataType();
    template <class T>
    static void ONNX_packTensor(onnx::TensorProto* onnxTensor,
                                const Tensor<T>& tensor,
                                const std::vector<size_t>& shape
                                                    = std::vector<size_t>());

    inline static std::unique_ptr<ONNX_CellExport> getInstance(Cell& cell);

    virtual ~ONNX_CellExport() {}

    // Commun methods for all cells
    virtual void generateNode(onnx::GraphProto* graph,
                                 const DeepNet& deepNet,
                                 const Cell& cell) = 0;
    virtual bool generateActivation(onnx::GraphProto* graph,
                                    const Cell& cell);
};
}

std::unique_ptr<N2D2::ONNX_CellExport> N2D2::ONNX_CellExport::getInstance(Cell& cell)
{
    return Registrar<ONNX_CellExport>::create(cell.getType())(cell);
}

template <class T>
onnx::TensorProto::DataType N2D2::ONNX_CellExport::ONNX_dataType()
{
    throw std::runtime_error("ONNX_CellExport::ONNX_dataType(): "
                             "tensor type not supported by ONNX!");
}

template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<float>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<half_float::half>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<double>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<int8_t>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<uint8_t>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<int16_t>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<uint16_t>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<int32_t>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<uint32_t>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<int64_t>();
template <>
onnx::TensorProto::DataType N2D2::ONNX_CellExport
    ::ONNX_dataType<uint64_t>();

template <class T>
void N2D2::ONNX_CellExport::ONNX_packTensor(
    onnx::TensorProto* onnxTensor,
    const Tensor<T>& tensor,
    const std::vector<size_t>& shape)
{
    onnxTensor->set_data_type(ONNX_dataType<T>());

    std::vector<size_t> dims = (!shape.empty()) ? shape : tensor.dims();
    std::reverse(dims.begin(), dims.end());
    std::for_each(dims.begin(), dims.end(), [&onnxTensor](size_t dim)
        { onnxTensor->mutable_dims()->Add(dim); });
    onnxTensor->set_raw_data(&tensor.data().data()[0],
        sizeof(T) * tensor.size());
}

#endif

#endif // N2D2_ONNX_CELLEXPORT_H
