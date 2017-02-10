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

#include "Export/CPP_cuDNN/CPP_cuDNN_SoftmaxCellExport.hpp"

N2D2::Registrar<N2D2::SoftmaxCellExport>
N2D2::CPP_cuDNN_SoftmaxCellExport::mRegistrar(
    "CPP_cuDNN", N2D2::CPP_cuDNN_SoftmaxCellExport::generate);

N2D2::Registrar<N2D2::CPP_cuDNN_CellExport>
N2D2::CPP_cuDNN_SoftmaxCellExport::mRegistrarType(
    SoftmaxCell::Type, N2D2::CPP_cuDNN_SoftmaxCellExport::getInstance);

void N2D2::CPP_cuDNN_SoftmaxCellExport::generate(SoftmaxCell& cell,
                                                 const std::string& dirName)
{
    CPP_SoftmaxCellExport::generate(cell, dirName);

}

std::unique_ptr<N2D2::CPP_cuDNN_SoftmaxCellExport>
N2D2::CPP_cuDNN_SoftmaxCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_cuDNN_SoftmaxCellExport>(new CPP_cuDNN_SoftmaxCellExport);
}

void N2D2::CPP_cuDNN_SoftmaxCellExport::generateCellProgramDesc(Cell& cell,
                                                                std::ofstream
                                                                & prog)
{

    generateCellProgramTensorDesc(cell, prog);
}

void N2D2::CPP_cuDNN_SoftmaxCellExport::generateCellProgramTensorDesc(
    Cell& cell, std::ofstream& prog)
{
    prog << "cudnnTensorDescriptor_t " << cell.getName()
         << "_tensorDescIn;\n"
            "cudnnTensorDescriptor_t " << cell.getName() << "_tensorDescOut;\n";
}

void N2D2::CPP_cuDNN_SoftmaxCellExport::generateCellProgramGlobalDefinition(
    Cell& /*cell*/, std::ofstream& /*prog*/)
{
}

void N2D2::CPP_cuDNN_SoftmaxCellExport::generateCellBuffer(const std::string
                                                           & bufferName,
                                                           std::ofstream& prog)
{
    prog << "std::vector<DATA_T *> " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_SoftmaxCellExport::generateCellProgramInitNetwork(
    Cell& cell, std::vector<std::string>& /*parentsName*/, std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << "    cudnnCreateTensorDescriptor(&"
        << cell.getName() << "_tensorDescIn);\n"
        << "    cudnnCreateTensorDescriptor(&"
        << cell.getName() << "_tensorDescOut);\n";

    prog << "    setSoftmax(batchSize,\n"
        << "                " << prefix << "_NB_CHANNELS,\n"
        << "                " << prefix << "_CHANNELS_HEIGHT,\n"
        << "                " << prefix << "_CHANNELS_WIDTH,\n"
        << "                " << "context_tensorFormat,\n"
        << "                " << "context_dataType,\n"
        << "                " << cell.getName() << "_tensorDescIn,\n"
        << "                " << cell.getName() << "_tensorDescOut);\n\n";
}

void N2D2::CPP_cuDNN_SoftmaxCellExport::generateCellProgramInitBuffer(
    Cell& cell, const std::string& bufferName, std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << bufferName << "buffer[" << prefix
        << "_OUTPUT_OFFSET], sizeof(DATA_T)*"
        << Utils::upperCase(cell.getName())
        << "_OUTPUTS_SIZE*batchSize));\n\n\n";
}

void N2D2::CPP_cuDNN_SoftmaxCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& output_pos,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string prefix = Utils::upperCase(cell.getName());
    const std::string proto = (funcProto.empty()) ? "    softmax" : funcProto;

    prog << proto
        << "( "
        << "                " << "context_handle,\n"
        << "                " << cell.getName() + "_tensorDescIn,\n"
        << "                " << inputName + ",\n"
        << "                " << cell.getName() + "_tensorDescOut,\n"
        << "                " << "(DATA_T**)&" + outputName + "["
                              <<  output_pos + "]);\n";
}

void N2D2::CPP_cuDNN_SoftmaxCellExport::generateCellProgramOutputFunction(
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

void N2D2::CPP_cuDNN_SoftmaxCellExport::generateCellProgramFree(Cell& cell,
    std::vector<std::string>& /*parentsName*/, std::ofstream& prog)
{
    prog << " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName()
         << "_tensorDescIn) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName() << "_tensorDescOut) );\n"
                              "\n";
}
