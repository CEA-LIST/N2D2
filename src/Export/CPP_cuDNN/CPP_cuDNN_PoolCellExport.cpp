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
    prog << "std::vector<cudnnTensorDescriptor_t> " << cell.getName()
        << "_tensorDescIn;\n"
        "std::vector<cudnnTensorDescriptor_t> " << cell.getName()
        << "_tensorDescOut;\n";
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
    prog << "std::vector<DATA_T *> " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramInitNetwork(
    Cell& cell, std::vector<std::string>& parentsName, std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    unsigned int parentSize = parentsName.size();
    prog << "    std::vector<int> " << cell.getName()
        << "_nbChanPerLayer;\n";
    prog << "    std::vector<int> " << cell.getName()
        << "_chanHeightPerLayer;\n";
    prog << "    std::vector<int> " << cell.getName()
        << "_chanWidthPerLayer;\n";

    for(unsigned int k = 0; k < parentSize; ++k) {
        const std::string prefixParent = Utils::upperCase(parentsName[k]);

        prog << "    " << cell.getName() << "_nbChanPerLayer.push_back("
            << prefixParent << "_NB_OUTPUTS);\n";
        prog << "    " << cell.getName() << "_chanHeightPerLayer.push_back("
            << prefixParent << "_OUTPUTS_HEIGHT);\n";
        prog << "    " << cell.getName() << "_chanWidthPerLayer.push_back("
            << prefixParent << "_OUTPUTS_WIDTH);\n";

    }

    prog << "    setPooling(batchSize,\n"
        << "                " << cell.getName() << "_nbChanPerLayer,\n"
        << "                " << cell.getName() << "_chanHeightPerLayer,\n"
        << "                " << cell.getName() << "_chanWidthPerLayer,\n"
        << "                " << prefix << "_PADDING_Y,\n"
        << "                " << prefix << "_PADDING_X,\n"
        << "                " << prefix << "_STRIDE_Y,\n"
        << "                " << prefix << "_STRIDE_X,\n"
        << "                " << prefix << "_OUTPUTS_HEIGHT,\n"
        << "                " << prefix << "_OUTPUTS_WIDTH,\n"
        << "                " << "context_tensorFormat,\n"
        << "                " << "context_dataType,\n"
        << "                " << cell.getName() << "_tensorDescIn,\n"
        << "                " << cell.getName() << "_tensorDescOut,\n"
        << "                " << prefix << "_POOLING,\n"
        << "                " << cell.getName() << "_pooling_cudnn,\n"
        << "                " << prefix << "_NB_OUTPUTS,\n"
        << "                " << prefix << "_POOL_HEIGHT,\n"
        << "                " << prefix << "_POOL_WIDTH,\n"
        << "                " << cell.getName() <<  "_poolingDesc\n"
        << "    " <<");\n\n\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramInitBuffer(
    Cell& cell, const std::string& bufferName, std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << bufferName << "buffer[" << prefix
        << "_OUTPUT_OFFSET], sizeof(DATA_T)*"
        << Utils::upperCase(cell.getName())
        << "_OUTPUTS_SIZE*batchSize));\n\n\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& output_pos,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string prefix = Utils::upperCase(cell.getName());
    const std::string proto = (funcProto.empty()) ? "    poolcell" : funcProto;

    prog << proto
        << "(\n"
        << "                " << "context_handle,\n"
        << "                " << cell.getName() + "_tensorDescIn,\n"
        << "                " << inputName + ",\n"
        << "                " << cell.getName() + "_tensorDescOut,\n"
        << "                " << "(DATA_T**)&" + outputName + "["
                              << output_pos + "],\n"
        << "                " << cell.getName() + "_poolingDesc\n"
        << "    " <<");\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramOutputFunction(
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

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramFree(Cell& cell,
    std::vector<std::string>& parentsName, std::ofstream& prog)
{
    for(int k = parentsName.size() - 1; k >= 0; --k) {

        prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << cell.getName()
            << "_tensorDescIn[" << k << "]) );\n";

        prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << cell.getName()
            << "_tensorDescOut[" << k << "]) );\n";

    }

    prog << "    CHECK_CUDNN_STATUS( cudnnDestroyPoolingDescriptor("
        << cell.getName() << "_poolingDesc) );\n"
            "\n";
}
