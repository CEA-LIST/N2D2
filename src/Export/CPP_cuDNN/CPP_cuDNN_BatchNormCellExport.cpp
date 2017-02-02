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
    Utils::createDirectories(dirName + "/include");

    const std::string fileName = dirName + "/include/" + cell.getName()
                                 + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    C_CellExport::generateHeaderBegin(cell, header);
    CPP_cuDNN_CellExport::generateHeaderIncludes(cell, header);
    CPP_OpenCL_BatchNormCellExport::generateHeaderConstants(cell, header);
    CPP_OpenCL_BatchNormCellExport::generateHeaderFreeParameters(cell, header);
    C_CellExport::generateHeaderEnd(cell, header);
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
    prog << "cudnnTensorDescriptor_t " << cell.getName()
         << "_tensorDescIn;\n"
            "cudnnTensorDescriptor_t " << cell.getName() << "_tensorDescOut;\n";
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramBatchNormDesc(
    Cell& cell, std::ofstream& prog)
{

    prog << "cudnnTensorDescriptor_t " << cell.getName() << "_scaleDesc;\n";
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{
    prog << "DATA_T* " << cell.getName() << "_scale_cudnn(NULL);\n"
            "DATA_T* " << cell.getName() << "_bias_cudnn(NULL);\n"
            "DATA_T* " << cell.getName() << "_mean_cudnn(NULL);\n"
            "DATA_T* " << cell.getName() << "_variance_cudnn(NULL);\n"
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
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << "    cudnnCreateTensorDescriptor(&" << cell.getName()
        << "_tensorDescIn);\n"
        << "    cudnnCreateTensorDescriptor(&" << cell.getName()
        << "_tensorDescOut);\n"
        << "    cudnnCreateTensorDescriptor(&" << cell.getName()
        << "_scaleDesc);\n";

    prog << "    setBatchnorm(batchSize,\n"
        << "                " << prefix << "_NB_CHANNELS,\n"
        << "                " << prefix << "_CHANNELS_HEIGHT,\n"
        << "                " << prefix << "_CHANNELS_WIDTH,\n"
        << "                " << cell.getName() << "_scaleDesc,\n"
        << "                " << "context_tensorFormat,\n"
        << "                " << "context_dataType,\n"
        << "                " << cell.getName() << "_tensorDescIn,\n"
        << "                " << cell.getName() << "_tensorDescOut);\n\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << cell.getName() << "_scale_cudnn, sizeof(DATA_T)*"
        << prefix
        << "_NB_OUTPUTS));\n";

    prog << "    CHECK_CUDA_STATUS( cudaMemcpy("
        << cell.getName() << "_scale_cudnn, "
        << cell.getName() << "_scales, "
        << prefix << "_NB_OUTPUTS*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << cell.getName() << "_bias_cudnn, sizeof(DATA_T)*"
        << prefix
        << "_NB_OUTPUTS));\n";

    prog << "    CHECK_CUDA_STATUS( cudaMemcpy("
        << cell.getName() << "_bias_cudnn, "
        << cell.getName() << "_biases, "
        << prefix << "_NB_OUTPUTS*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << cell.getName() << "_variance_cudnn, sizeof(DATA_T)*"
        << prefix
        << "_NB_OUTPUTS));\n";

    prog << "    CHECK_CUDA_STATUS( cudaMemcpy("
        << cell.getName() << "_variance_cudnn, "
        << cell.getName() << "_variances, "
        << prefix << "_NB_OUTPUTS*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << cell.getName() << "_mean_cudnn, sizeof(DATA_T)*"
        << prefix
        << "_NB_OUTPUTS));\n";

    prog << "    CHECK_CUDA_STATUS( cudaMemcpy("
        << cell.getName() << "_mean_cudnn, "
        << cell.getName() << "_means, "
        << prefix << "_NB_OUTPUTS*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n";
}

void N2D2::CPP_cuDNN_BatchNormCellExport::generateCellProgramInitBuffer(
    Cell& cell, const std::string& bufferName, std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << bufferName << "buffer[" << prefix
        << "_OUTPUT_OFFSET], sizeof(DATA_T)*"
        << Utils::upperCase(cell.getName())
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
    const std::string prefix = Utils::upperCase(cell.getName());
    const std::string proto =
        (funcProto.empty()) ? "    batchnormcell"
        : funcProto;

    prog << proto
        << "(\n"
        << "                " << "context_handle,\n"
        << "                " << "batchSize,\n"
        << "                " << prefix + "_NB_CHANNELS,\n"
        << "                " << prefix + "_CHANNELS_HEIGHT,\n"
        << "                " << prefix + "_CHANNELS_WIDTH,\n"
        << "                " << cell.getName() + "_tensorDescIn,\n"
        << "                " << inputName + ",\n"
        << "                " << cell.getName() + "_scale_cudnn,\n"
        << "                " << cell.getName() + "_scaleDesc,\n"
        << "                " << cell.getName() + "_bias_cudnn,\n"
        << "                " << cell.getName() + "_mean_cudnn,\n"
        << "                " << cell.getName() + "_variance_cudnn,\n"
        << "                " << prefix + "_EPSILON,\n"
        << "                " << cell.getName() + "_tensorDescOut,\n"
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
    const std::string prefix = Utils::upperCase(cell.getName());

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
    prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << cell.getName() << "_scaleDesc) );\n"
            "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << cell.getName() << "_tensorDescIn) );\n"
            "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << cell.getName() << "_tensorDescOut) );\n"
            "    CHECK_CUDA_STATUS( cudaFree("
            << cell.getName() << "_bias_cudnn) );\n"
            "    CHECK_CUDA_STATUS( cudaFree("
            << cell.getName() << "_scale_cudnn) );\n"
            "    CHECK_CUDA_STATUS( cudaFree("
            << cell.getName() << "_variance_cudnn) );\n"
            "    CHECK_CUDA_STATUS( cudaFree("
            << cell.getName() << "_mean_cudnn) );\n"
            "\n";
}
