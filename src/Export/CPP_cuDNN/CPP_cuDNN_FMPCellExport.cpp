
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

#include "Export/CPP_cuDNN/CPP_cuDNN_FMPCellExport.hpp"

N2D2::Registrar<N2D2::FMPCellExport> N2D2::CPP_cuDNN_FMPCellExport::mRegistrar(
    "CPP_cuDNN", N2D2::CPP_cuDNN_FMPCellExport::generate);

N2D2::Registrar<N2D2::CPP_cuDNN_CellExport>
N2D2::CPP_cuDNN_FMPCellExport::mRegistrarType(
    FMPCell::Type, N2D2::CPP_cuDNN_FMPCellExport::getInstance);

void N2D2::CPP_cuDNN_FMPCellExport::generate(FMPCell& cell,
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
    CPP_OpenCL_FMPCellExport::generateHeaderConstants(cell, header);
    CPP_OpenCL_FMPCellExport::generateHeaderConnections(cell, header);
    CPP_OpenCL_FMPCellExport::generateHeaderGrid(cell, header);
    C_CellExport::generateHeaderEnd(cell, header);
}

std::unique_ptr<N2D2::CPP_cuDNN_FMPCellExport>
N2D2::CPP_cuDNN_FMPCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_cuDNN_FMPCellExport>(new CPP_cuDNN_FMPCellExport);
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramDesc(Cell& cell,
                                                            std::ofstream& prog)
{

    generateCellProgramTensorDesc(cell, prog);
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramTensorDesc(Cell& cell,
                                                                  std::ofstream
                                                                  & prog)
{
    prog << "cudnnTensorDescriptor_t " << cell.getName()
         << "_tensorDescIn;\n"
            "cudnnTensorDescriptor_t " << cell.getName() << "_tensorDescOut;\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{

    prog << "unsigned int *" << cell.getName() << "_gridx_cudnn(NULL);\n"
                                                  "unsigned int *"
         << cell.getName() << "_gridy_cudnn(NULL);\n"
                              "\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellBuffer(const std::string
                                                       & bufferName,
                                                       std::ofstream& prog)
{
    prog << "DATA_T * " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramInitNetwork(Cell& cell,
                                                                   std::ofstream
                                                                   & prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_tensorDescIn);\n"
         << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_tensorDescOut);\n";

    prog << " setFmp();\n\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramInitBuffer(
    const std::string& bufferName, std::ofstream& prog)
{
    prog << " CHECK_CUDA_STATUS(cudaMalloc(&" << bufferName
         << "buffer, sizeof(DATA_T)*" << Utils::upperCase(bufferName)
         << "OUTPUTS_SIZE*batchSize));\n";

    prog << " CHECK_CUDA_STATUS(cudaMalloc(&" << bufferName
         << "gridx_cudnn, sizeof(unsigned int)*" << Utils::upperCase(bufferName)
         << "GRIDX_SIZE));\n"
         << " CHECK_CUDA_STATUS( cudaMemcpy(" << bufferName << "gridx_cudnn, "
         << bufferName << "gridx_flatten, " << Utils::upperCase(bufferName)
         << "GRIDX_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice) );\n"
            "\n";

    prog << " CHECK_CUDA_STATUS(cudaMalloc(&" << bufferName
         << "gridy_cudnn, sizeof(unsigned int)*" << Utils::upperCase(bufferName)
         << "GRIDY_SIZE));\n"
         << " CHECK_CUDA_STATUS( cudaMemcpy(" << bufferName << "gridy_cudnn, "
         << bufferName << "gridy_flatten, " << Utils::upperCase(bufferName)
         << "GRIDY_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice) );\n"
            "\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string prefix = Utils::upperCase(cell.getName());
    const std::string proto = (funcProto.empty()) ? " fmpcell" : funcProto;

    prog << proto << "( "
         << "context_handle, batchSize," << prefix + "_NB_CHANNELS,"
         << prefix + "_CHANNELS_HEIGHT," << prefix + "_CHANNELS_WIDTH,"
         << cell.getName() + "_gridx_cudnn, "
         << cell.getName() + "_gridy_cudnn, " << prefix + "_OVERLAPPING,"
         << inputName + ", " << prefix + "_OUTPUTS_SIZE,"
         << prefix + "_OUTPUTS_HEIGHT," << prefix + "_OUTPUTS_WIDTH,"
         << prefix + "_NB_OUTPUTS," << prefix + "_OUTPUT_OFFSET*batchSize, "
         << outputName + "); \n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramOutputFunction(
    Cell& cell,
    const std::string& outputDataName,
    const std::string& outputName,
    std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    if ((cell.getOutputsWidth() == 1) && (cell.getOutputsHeight() == 1)) {
        prog << " output_generation(batchSize, " << prefix << "_NB_OUTPUTS, "
             << outputDataName << ", " << outputName << ");\n";
    } else {
        prog << " spatial_output_generation(batchSize, " << prefix
             << "_NB_OUTPUTS, " << prefix << "_OUTPUTS_HEIGHT, " << prefix
             << "_OUTPUTS_WIDTH, " << outputDataName << ", " << outputName
             << ");\n";
    }
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramFree(Cell& cell,
                                                            std::ofstream& prog)
{
    prog << " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName()
         << "_tensorDescIn) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName() << "_tensorDescOut) );\n"
                              "\n";
}
