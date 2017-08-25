/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "Export/CPP_cuDNN/CPP_cuDNN_BatchNormCellExport.hpp"

N2D2::Registrar<N2D2::BatchNormCellExport>
N2D2::CPP_cuDNN_BatchNormCellExport::mRegistrar(
    "CPP_cuDNN", N2D2::CPP_cuDNN_BatchNormCellExport::generate);

N2D2::Registrar<N2D2::CPP_cuDNN_CellExport>
N2D2::CPP_cuDNN_BatchNormCellExport::mRegistrarType(
    BatchNormCell::Type, N2D2::CPP_cuDNN_BatchNormCellExport::getInstance);

void N2D2::CPP_cuDNN_BatchNormCellExport::generate(BatchNormCell& cell,
                                                   const std::string& dirName)
{
    N2D2::CPP_BatchNormCellExport::generate(cell, dirName);
}

std::unique_ptr<N2D2::CPP_cuDNN_BatchNormCellExport>
N2D2::CPP_cuDNN_BatchNormCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_cuDNN_BatchNormCellExport>(new CPP_cuDNN_BatchNormCellExport);
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramDesc(Cell& cell,
                                                                  std::ofstream
                                                                  & prog)
{

    generateCellProgramTensorDesc(cell, prog);
    generateCellProgramBatchNormDesc(cell, prog);
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramTensorDesc(
    Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "cudnnTensorDescriptor_t " << identifier
         << "_tensorDescIn;\n"
            "cudnnTensorDescriptor_t " << identifier << "_tensorDescOut;\n";
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramBatchNormDesc(
    Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "cudnnTensorDescriptor_t " << identifier << "_scaleDesc;\n";
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "DATA_T* " << identifier << "_scale_cudnn(NULL);\n"
            "DATA_T* " << identifier << "_bias_cudnn(NULL);\n"
            "DATA_T* " << identifier << "_mean_cudnn(NULL);\n"
            "DATA_T* " << identifier << "_variance_cudnn(NULL);\n"
            "\n";

}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellBuffer(const std::string
                                                             & bufferName,
                                                             std::ofstream
                                                             & prog)
{
    prog << "std::vector<DATA_T *> " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramInitNetwork(
    Cell& cell, std::vector<std::string>& /*parentsName*/, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "    cudnnCreateTensorDescriptor(&" << identifier
        << "_tensorDescIn);\n"
        << "    cudnnCreateTensorDescriptor(&" << identifier
        << "_tensorDescOut);\n"
        << "    cudnnCreateTensorDescriptor(&" << identifier
        << "_scaleDesc);\n";

    prog << "    setBatchnorm(batchSize,\n"
        << "                " << prefix << "_NB_CHANNELS,\n"
        << "                " << prefix << "_CHANNELS_HEIGHT,\n"
        << "                " << prefix << "_CHANNELS_WIDTH,\n"
        << "                " << identifier << "_scaleDesc,\n"
        << "                " << "context_tensorFormat,\n"
        << "                " << "context_dataType,\n"
        << "                " << identifier << "_tensorDescIn,\n"
        << "                " << identifier << "_tensorDescOut);\n\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << identifier << "_scale_cudnn, sizeof(DATA_T)*"
        << prefix
        << "_NB_OUTPUTS));\n";

    prog << "    CHECK_CUDA_STATUS( cudaMemcpy("
        << identifier << "_scale_cudnn, "
        << identifier << "_scales, "
        << prefix << "_NB_OUTPUTS*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << identifier << "_bias_cudnn, sizeof(DATA_T)*"
        << prefix
        << "_NB_OUTPUTS));\n";

    prog << "    CHECK_CUDA_STATUS( cudaMemcpy("
        << identifier << "_bias_cudnn, "
        << identifier << "_biases, "
        << prefix << "_NB_OUTPUTS*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << identifier << "_variance_cudnn, sizeof(DATA_T)*"
        << prefix
        << "_NB_OUTPUTS));\n";

    prog << "    CHECK_CUDA_STATUS( cudaMemcpy("
        << identifier << "_variance_cudnn, "
        << identifier << "_variances, "
        << prefix << "_NB_OUTPUTS*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << identifier << "_mean_cudnn, sizeof(DATA_T)*"
        << prefix
        << "_NB_OUTPUTS));\n";

    prog << "    CHECK_CUDA_STATUS( cudaMemcpy("
        << identifier << "_mean_cudnn, "
        << identifier << "_means, "
        << prefix << "_NB_OUTPUTS*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n";
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramInitBuffer(
    Cell& cell, const std::string& bufferName, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << bufferName << "buffer[" << prefix
        << "_OUTPUT_OFFSET], sizeof(DATA_T)*"
        << prefix
        << "_OUTPUTS_SIZE*batchSize));\n\n\n";
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& output_pos,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto =
        (funcProto.empty()) ? "    batchnormcell"
        : funcProto;

    prog << proto
        << "(\n"
        << "                " << "CudaContext::cudnnHandle(),\n"
        << "                " << "batchSize,\n"
        << "                " << prefix + "_NB_CHANNELS,\n"
        << "                " << prefix + "_CHANNELS_HEIGHT,\n"
        << "                " << prefix + "_CHANNELS_WIDTH,\n"
        << "                " << identifier + "_tensorDescIn,\n"
        << "                " << inputName + ",\n"
        << "                " << identifier + "_scale_cudnn,\n"
        << "                " << identifier + "_scaleDesc,\n"
        << "                " << identifier + "_bias_cudnn,\n"
        << "                " << identifier + "_mean_cudnn,\n"
        << "                " << identifier + "_variance_cudnn,\n"
        << "                " << prefix + "_EPSILON,\n"
        << "                " << identifier + "_tensorDescOut,\n"
        << "                " << "(DATA_T**)&" + outputName + "["
                              << output_pos + "],\n"
        << "                " << prefix + "_ACTIVATION); \n";
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramOutputFunction(
    Cell& cell,
    const std::string& outputDataName,
    const std::string& outputName,
    std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                            cell.getName()));

    if( (cell.getOutputsWidth() == 1) && (cell.getOutputsHeight() == 1) ){
        prog << "    output_generation(batchSize, "
            << prefix << "_NB_OUTPUTS, "
            << outputDataName << ", "
            << outputName << ");\n";
    }
    else {
        prog << "    spatial_output_generation(batchSize, "
            << prefix << "_NB_OUTPUTS, "
            << prefix << "_OUTPUTS_HEIGHT, "
            << prefix << "_OUTPUTS_WIDTH, "
            << outputDataName << ", "
            << outputName << ");\n";
    }
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramFree(
    Cell& cell, std::vector<std::string>& /*parentsName*/, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << identifier << "_scaleDesc) );\n"
            "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << identifier << "_tensorDescIn) );\n"
            "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << identifier << "_tensorDescOut) );\n"
            "    CHECK_CUDA_STATUS( cudaFree("
            << identifier << "_bias_cudnn) );\n"
            "    CHECK_CUDA_STATUS( cudaFree("
            << identifier << "_scale_cudnn) );\n"
            "    CHECK_CUDA_STATUS( cudaFree("
            << identifier << "_variance_cudnn) );\n"
            "    CHECK_CUDA_STATUS( cudaFree("
            << identifier << "_mean_cudnn) );\n"
            "\n";
}
