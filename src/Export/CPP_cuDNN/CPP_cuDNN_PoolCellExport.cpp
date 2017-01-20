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

#include "Export/CPP_cuDNN/CPP_cuDNN_PoolCellExport.hpp"

N2D2::Registrar<N2D2::PoolCellExport>
N2D2::CPP_cuDNN_PoolCellExport::mRegistrar(
    "CPP_cuDNN", N2D2::CPP_cuDNN_PoolCellExport::generate);

N2D2::Registrar<N2D2::CPP_cuDNN_CellExport>
N2D2::CPP_cuDNN_PoolCellExport::mRegistrarType(
    PoolCell::Type, N2D2::CPP_cuDNN_PoolCellExport::getInstance);

void N2D2::CPP_cuDNN_PoolCellExport::generate(PoolCell& cell,
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
    generateHeaderConstants(cell, header);
    generateHeaderConnections(cell, header);
    C_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_cuDNN_PoolCellExport::generateHeaderConstants(PoolCell& cell,
                                                             std::ofstream
                                                             & header)
{
    C_PoolCellExport::generateHeaderConstants(cell, header);

    const std::string prefix = Utils::upperCase(cell.getName());

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_OUTPUTS_WIDTH*" << prefix
           << "_OUTPUTS_HEIGHT)\n"
              "#define " << prefix << "_CHANNELS_SIZE (" << prefix
           << "_NB_CHANNELS*" << prefix << "_CHANNELS_WIDTH*" << prefix
           << "_CHANNELS_HEIGHT)\n"
              "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix
           << "_OUTPUTS_SIZE, " << prefix << "_CHANNELS_SIZE))\n\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateHeaderConnections(PoolCell& cell,
                                                               std::ofstream
                                                               & header)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    header << "#define " << prefix << "_MAPPING_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n"
           << "static char " << cell.getName() << "_mapping_flatten[" << prefix
           << "_MAPPING_SIZE] = {\n";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (output > 0 || channel > 0)
                header << ", ";

            if (!cell.isConnection(channel, output))
                header << "0";
            else
                header << "1";
        }
    }

    header << "};\n\n";
}

std::unique_ptr<N2D2::CPP_cuDNN_PoolCellExport>
N2D2::CPP_cuDNN_PoolCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_cuDNN_PoolCellExport>(new CPP_cuDNN_PoolCellExport);
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramDesc(Cell& cell,
                                                             std::ofstream
                                                             & prog)
{

    generateCellProgramTensorDesc(cell, prog);
    generateCellProgramPoolDesc(cell, prog);
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramTensorDesc(Cell& cell,
                                                                   std::ofstream
                                                                   & prog)
{
    prog << "cudnnTensorDescriptor_t " << cell.getName()
         << "_tensorDescIn;\n"
            "cudnnTensorDescriptor_t " << cell.getName() << "_tensorDescOut;\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramPoolDesc(Cell& cell,
                                                                 std::ofstream
                                                                 & prog)
{

    prog << "cudnnPoolingDescriptor_t " << cell.getName() << "_poolingDesc;\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{

    prog << "cudnnPoolingMode_t " << cell.getName() << "_pooling_cudnn;\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellBuffer(const std::string
                                                        & bufferName,
                                                        std::ofstream& prog)
{
    prog << "DATA_T * " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramInitNetwork(
    Cell& cell, std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_tensorDescIn);\n"
         << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_tensorDescOut);\n";

    prog << " cudnnCreatePoolingDescriptor(&" << cell.getName()
         << "_poolingDesc);\n";

    prog << " setPooling(batchSize, " << prefix << "_NB_CHANNELS, " << prefix
         << "_CHANNELS_HEIGHT, " << prefix << "_CHANNELS_WIDTH, " << prefix
         << "_STRIDE_Y, " << prefix << "_STRIDE_X, " << prefix
         << "_OUTPUTS_HEIGHT, " << prefix
         << "_OUTPUTS_WIDTH, "
            "context_tensorFormat, context_dataType," << cell.getName()
         << "_tensorDescIn, " << cell.getName() << "_tensorDescOut, " << prefix
         << "_POOLING, " << cell.getName() << "_pooling_cudnn, " << prefix
         << "_NB_OUTPUTS, " << prefix << "_OUTPUT_OFFSET*batchSize, " << prefix
         << "_POOL_HEIGHT, " << prefix << "_POOL_WIDTH, " << cell.getName()
         << "_poolingDesc);\n\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramInitBuffer(
    const std::string& bufferName, std::ofstream& prog)
{
    prog << " CHECK_CUDA_STATUS(cudaMalloc(&" << bufferName
         << "buffer, sizeof(DATA_T)*" << Utils::upperCase(bufferName)
         << "OUTPUTS_SIZE*batchSize));\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string prefix = Utils::upperCase(cell.getName());
    const std::string proto = (funcProto.empty()) ? " poolcell" : funcProto;

    prog << proto << "( "
         << "context_handle, "
         << cell.getName() + "_tensorDescIn, " + inputName + ", "
         << prefix + "_OUTPUT_OFFSET*batchSize, "
         << cell.getName() + "_tensorDescOut, (DATA_T**)&" + outputName + ", "
         << cell.getName() + "_poolingDesc); \n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramOutputFunction(
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

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramFree(Cell& cell,
                                                             std::ofstream
                                                             & prog)
{
    prog << " CHECK_CUDNN_STATUS( cudnnDestroyPoolingDescriptor("
         << cell.getName()
         << "_poolingDesc) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName()
         << "_tensorDescIn) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName() << "_tensorDescOut) );\n"
                              "\n";
}
