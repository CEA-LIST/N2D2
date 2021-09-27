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

    static void ONNX_castAndPackTensor(onnx::TensorProto* onnxTensor,
                                       const BaseTensor& tensor,
                                       const std::vector<size_t>& shape
                                                    = std::vector<size_t>());
    static void ONNX_packTensor(onnx::TensorProto* onnxTensor,
                                const BaseTensor& tensor,
                                const std::vector<size_t>& shape
                                                    = std::vector<size_t>());
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

namespace N2D2 {
template <>
void ONNX_CellExport::ONNX_packTensor<float>(
    onnx::TensorProto* onnxTensor,
    const Tensor<float>& tensor,
    const std::vector<size_t>& shape);

template <>
void ONNX_CellExport::ONNX_packTensor<half_float::half>(
    onnx::TensorProto* onnxTensor,
    const Tensor<half_float::half>& tensor,
    const std::vector<size_t>& shape);

template <>
void ONNX_CellExport::ONNX_packTensor<double>(
    onnx::TensorProto* onnxTensor,
    const Tensor<double>& tensor,
    const std::vector<size_t>& shape);

template <>
void ONNX_CellExport::ONNX_packTensor<int8_t>(
    onnx::TensorProto* onnxTensor,
    const Tensor<int8_t>& tensor,
    const std::vector<size_t>& shape);

template <>
void ONNX_CellExport::ONNX_packTensor<int16_t>(
    onnx::TensorProto* onnxTensor,
    const Tensor<int16_t>& tensor,
    const std::vector<size_t>& shape);

template <>
void ONNX_CellExport::ONNX_packTensor<int32_t>(
    onnx::TensorProto* onnxTensor,
    const Tensor<int32_t>& tensor,
    const std::vector<size_t>& shape);
}

#endif

#endif // N2D2_ONNX_CELLEXPORT_H

