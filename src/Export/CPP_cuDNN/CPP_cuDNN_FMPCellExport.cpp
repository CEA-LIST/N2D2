
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
    N2D2::CPP_FMPCellExport::generate(cell, dirName);
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

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramTensorDesc(
    Cell& /*cell*/, std::ofstream& /*prog*/)
{

}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "unsigned int *" << identifier << "_gridx_cudnn(NULL);\n"
                                                  "unsigned int *"
         << identifier << "_gridy_cudnn(NULL);\n"
                              "\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellBuffer(const std::string
                                                       & bufferName,
                                                       std::ofstream& prog)
{
    prog << "std::vector<DATA_T *> " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramInitNetwork(Cell& /*cell*/,
    std::vector<std::string>& /*parentsName*/, std::ofstream& prog)
{
    prog << " setFmp();\n\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramInitBuffer(Cell& cell,
    const std::string& bufferName, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << bufferName << "buffer[" << prefix
        << "_OUTPUT_OFFSET], sizeof(DATA_T)*"
        << prefix
        << "_OUTPUTS_SIZE*batchSize));\n\n\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << bufferName << "gridx_cudnn, sizeof(unsigned int)*"
        << Utils::upperCase(bufferName)
        << "GRIDX_SIZE));\n"
        << " CHECK_CUDA_STATUS( cudaMemcpy("
        << bufferName << "gridx_cudnn, "
        << bufferName << "gridx_flatten, "
        << Utils::upperCase(bufferName)
        << "GRIDX_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice) );\n"
        "\n";

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << bufferName << "gridy_cudnn, sizeof(unsigned int)*"
        << Utils::upperCase(bufferName)
        << "GRIDY_SIZE));\n"
        << " CHECK_CUDA_STATUS( cudaMemcpy("
        << bufferName << "gridy_cudnn, "
        << bufferName << "gridy_flatten, "
        << Utils::upperCase(bufferName)
        << "GRIDY_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice) );\n"
        "\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& output_pos,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto = (funcProto.empty()) ? "    fmpcell" : funcProto;

    prog << proto
        << "(\n"
        << "                " << "CudaContext::cudnnHandle(),\n"
        << "                " << "batchSize,\n"
        << "                " << prefix + "_NB_CHANNELS,\n"
        << "                " << prefix + "_CHANNELS_HEIGHT,\n"
        << "                " << prefix + "_CHANNELS_WIDTH,\n"
        << "                " << identifier + "_gridx_cudnn,\n"
        << "                " << identifier + "_gridy_cudnn,\n"
        << "                " << prefix + "_OVERLAPPING,\n"
        << "                " << inputName + ",\n"
        << "                " << prefix + "_OUTPUTS_SIZE,\n"
        << "                " << prefix + "_OUTPUTS_HEIGHT,\n"
        << "                " << prefix + "_OUTPUTS_WIDTH,\n"
        << "                " << prefix + "_NB_OUTPUTS,\n"
        << "                " << prefix + "_OUTPUT_OFFSET*batchSize,\n"
        << "                " << outputName  + "["
                              << output_pos + "]);\n";
}

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramOutputFunction(
    Cell& cell,
    const std::string& outputDataName,
    const std::string& outputName,
    std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

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

void N2D2::CPP_cuDNN_FMPCellExport::generateCellProgramFree(Cell& cell,
    std::vector<std::string>& /*parentsName*/, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "    CHECK_CUDA_STATUS( cudaFree(" << identifier
            << "_gridy_cudnn) );\n"
        << "    CHECK_CUDA_STATUS( cudaFree(" << identifier
            << "_gridx_cudnn) );\n"
        << "\n";
}
