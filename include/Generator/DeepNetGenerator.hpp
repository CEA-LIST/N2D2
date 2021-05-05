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

#ifndef N2D2_DEEPNETGENERATOR_H
#define N2D2_DEEPNETGENERATOR_H

#include <memory>
#include <string>

#ifdef ONNX
#include <onnx.pb.h>
#endif

namespace N2D2 {

class DeepNet;
class Network;
class IniParser;

class DeepNetGenerator {
public:
    static std::shared_ptr<DeepNet> generate(Network& network,
                                             const std::string& fileName);
    static std::shared_ptr<DeepNet> generateFromINI(Network& network,
                                                   const std::string& fileName);
#ifdef ONNX
    static std::shared_ptr<DeepNet> generateFromONNX(Network& network,
        const std::string& fileName,
        IniParser& iniConfig,
        std::shared_ptr<DeepNet> deepNet = std::shared_ptr<DeepNet>(),
        const std::vector<std::shared_ptr<Cell>>& parentCells
            = std::vector<std::shared_ptr<Cell>>());

private:
    static void ONNX_processGraph(std::shared_ptr<DeepNet> deepNet,
        const std::vector<std::shared_ptr<Cell> >& parentCells,
        const onnx::GraphProto& graph,
        int opsetVersion,
        IniParser& iniConfig);
    static std::shared_ptr<BaseTensor> ONNX_unpackTensor(const onnx::TensorProto* tensor,
                                       const std::vector<unsigned int>& expectedDims
                                         = std::vector<unsigned int>());
    template <class T>
    static Tensor<T> ONNX_unpackTensor(const onnx::TensorProto* tensor,
                                       const std::vector<unsigned int>& expectedDims
                                         = std::vector<unsigned int>());
#endif
};
}

#ifdef ONNX
template <class T>
N2D2::Tensor<T> N2D2::DeepNetGenerator::ONNX_unpackTensor(
    const onnx::TensorProto* onnxTensor,
    const std::vector<unsigned int>& expectedDims)
{
    Tensor<T> tensor(expectedDims);

    // Find out dimensions from TensorProto
    std::vector<unsigned int> size;
    for (int i = 0; i < onnxTensor->dims_size(); ++i)
        size.push_back(onnxTensor->dims(i));
    std::reverse(size.begin(), size.end());

    if (expectedDims.empty())
        tensor.resize(std::vector<size_t>(size.begin(), size.end()));
    else if (expectedDims != size) {
        std::ostringstream errorStr;
        errorStr << "Unexpected size for ONNX tensor \""
            << onnxTensor->name() << "\": expected " << expectedDims
            << " , got " << size << std::endl;

        throw std::runtime_error(errorStr.str());
    }

    const std::string dataTypeName = onnxTensor->GetTypeName();

    if ((onnxTensor->data_type() == onnx::TensorProto_DataType_FLOAT
            && !std::is_same<T, float>::value)
        || (onnxTensor->data_type() == onnx::TensorProto_DataType_DOUBLE
            && !std::is_same<T, double>::value)
        || (onnxTensor->data_type() == onnx::TensorProto_DataType_INT32
            && !std::is_same<T, int32_t>::value)
        || (onnxTensor->data_type() == onnx::TensorProto_DataType_INT64
            && !std::is_same<T, int64_t>::value)
        || (onnxTensor->data_type() == onnx::TensorProto_DataType_UINT64
            && !std::is_same<T, uint64_t>::value))
        //|| (onnxTensor->data_type() == onnx::TensorProto_DataType_STRING
        //    && !std::is_same<T, std::string>::value))
    {
        std::ostringstream errorStr;
        errorStr << "Unexpected type for ONNX tensor \""
            << onnxTensor->name() << "\": expected " << typeid(T).name()
            << " , got " << dataTypeName << std::endl;

        throw std::runtime_error(errorStr.str());
    }

    const int dataSize
        = (std::is_same<T, float>::value) ? onnxTensor->float_data_size()
        : (std::is_same<T, double>::value) ? onnxTensor->double_data_size()
        : (std::is_same<T, int32_t>::value) ? onnxTensor->int32_data_size()
        : (std::is_same<T, int64_t>::value) ? onnxTensor->int64_data_size()
        : (std::is_same<T, uint64_t>::value) ? onnxTensor->uint64_data_size()
        //: (std::is_same<T, std::string>::value) ? onnxTensor->string_data_size()
        : 0;

    if (dataSize > 0) {
        if (tensor.empty())
            tensor.resize({(size_t)dataSize});

        assert((int)tensor.size() == dataSize);

        for (size_t i = 0; i < (size_t)dataSize; ++i) {
            if (std::is_same<T, float>::value)
                tensor(i) = onnxTensor->float_data(i);
            else if (std::is_same<T, double>::value)
                tensor(i) = onnxTensor->double_data(i);
            else if (std::is_same<T, int32_t>::value)
                tensor(i) = onnxTensor->int32_data(i);
            else if (std::is_same<T, int64_t>::value)
                tensor(i) = onnxTensor->int64_data(i);
            else if (std::is_same<T, uint64_t>::value)
                tensor(i) = onnxTensor->uint64_data(i);
            // This cause conversion issue, should be properly fixed with
            // conditional template or constexpr...
            //else if (std::is_same<T, std::string>::value)
            //    tensor(i) = onnxTensor->string_data(i);
        }
    }
    else if (onnxTensor->raw_data().size() > 0) {
        if (tensor.empty())
            tensor.resize({onnxTensor->raw_data().size() / sizeof(T)});

        assert(onnxTensor->raw_data().size() == (int)tensor.size() * sizeof(T));

        memcpy(&(tensor.data())[0],
                onnxTensor->raw_data().c_str(),
                tensor.size() * sizeof(T));
    }
    else {
        /*std::ostringstream errorStr;
        errorStr << "Missing data for ONNX tensor \""
            << onnxTensor->name() << "\"" << std::endl;

        throw std::runtime_error(errorStr.str());*/
        std::cout << "  No data for ONNX tensor \"" << onnxTensor->name() 
            << "\"" << std::endl;
    }

    return tensor;
}
#endif

#endif // N2D2_DEEPNETGENERATOR_H
