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
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_cuDNN_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    generateHeaderConnections(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_cuDNN_PoolCellExport::generateHeaderConstants(PoolCell& cell,
                                                             std::ofstream
                                                             & header)
{
    // Constants
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs()
           << "\n"
              "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels()
           << "\n"
              "#define " << prefix << "_OUTPUTS_WIDTH "
           << cell.getOutputsWidth() << "\n"
                                        "#define " << prefix
           << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
                                                               "#define "
           << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth()
           << "\n"
              "#define " << prefix << "_CHANNELS_HEIGHT "
           << cell.getChannelsHeight() << "\n"
                                          "#define " << prefix << "_POOL_WIDTH "
           << cell.getPoolWidth() << "\n"
                                     "#define " << prefix << "_POOL_HEIGHT "
           << cell.getPoolHeight() << "\n"
                                      "#define " << prefix << "_PADDING_X "
           << cell.getPaddingX() << "\n"
                                      "#define " << prefix << "_PADDING_Y "
           << cell.getPaddingY() << "\n"
                                      "#define " << prefix << "_STRIDE_X "
           << cell.getStrideX() << "\n"
                                   "#define " << prefix << "_STRIDE_Y "
           << cell.getStrideY() << "\n"
                                   "#define " << prefix << "_POOLING "
           << cell.getPooling() << "\n\n";

    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);

    if (cellFrame != NULL) {
        header << "#define " << prefix << "_ACTIVATION "
               << ((cellFrame->getActivation())
                       ? cellFrame->getActivation()->getType()
                       : "Linear") << "\n";
    }

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
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_MAPPING_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n"
           << "static char " << identifier << "_mapping_flatten[" << prefix
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
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "std::vector<cudnnTensorDescriptor_t> " << identifier
        << "_tensorDescIn;\n"
        "std::vector<cudnnTensorDescriptor_t> " << identifier
        << "_tensorDescOut;\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramPoolDesc(Cell& cell,
                                                                 std::ofstream
                                                                 & prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "cudnnPoolingDescriptor_t " << identifier << "_poolingDesc;\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "cudnnPoolingMode_t " << identifier << "_pooling_cudnn;\n";
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
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    unsigned int parentSize = parentsName.size();
    prog << "    std::vector<int> " << identifier
        << "_nbChanPerLayer;\n";
    prog << "    std::vector<int> " << identifier
        << "_chanHeightPerLayer;\n";
    prog << "    std::vector<int> " << identifier
        << "_chanWidthPerLayer;\n";

    for(unsigned int k = 0; k < parentSize; ++k) {
        const std::string prefixParent = Utils::upperCase(parentsName[k]);

        prog << "    " << identifier << "_nbChanPerLayer.push_back("
            << prefixParent << "_NB_OUTPUTS);\n";
        prog << "    " << identifier << "_chanHeightPerLayer.push_back("
            << prefixParent << "_OUTPUTS_HEIGHT);\n";
        prog << "    " << identifier << "_chanWidthPerLayer.push_back("
            << prefixParent << "_OUTPUTS_WIDTH);\n";

    }

    prog << "    setPooling(batchSize,\n"
        << "                " << identifier << "_nbChanPerLayer,\n"
        << "                " << identifier << "_chanHeightPerLayer,\n"
        << "                " << identifier << "_chanWidthPerLayer,\n"
        << "                " << prefix << "_PADDING_Y,\n"
        << "                " << prefix << "_PADDING_X,\n"
        << "                " << prefix << "_STRIDE_Y,\n"
        << "                " << prefix << "_STRIDE_X,\n"
        << "                " << prefix << "_OUTPUTS_HEIGHT,\n"
        << "                " << prefix << "_OUTPUTS_WIDTH,\n"
        << "                " << "context_tensorFormat,\n"
        << "                " << "context_dataType,\n"
        << "                " << identifier << "_tensorDescIn,\n"
        << "                " << identifier << "_tensorDescOut,\n"
        << "                " << prefix << "_POOLING,\n"
        << "                " << identifier << "_pooling_cudnn,\n"
        << "                " << prefix << "_NB_OUTPUTS,\n"
        << "                " << prefix << "_POOL_HEIGHT,\n"
        << "                " << prefix << "_POOL_WIDTH,\n"
        << "                " << identifier <<  "_poolingDesc\n"
        << "    " <<");\n\n\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramInitBuffer(
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

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& output_pos,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto = (funcProto.empty()) ? "    poolcell" : funcProto;

    prog << proto
        << "(\n"
        << "                " << "CudaContext::cudnnHandle(),\n"
        << "                " << identifier + "_tensorDescIn,\n"
        << "                " << inputName + ",\n"
        << "                " << identifier + "_tensorDescOut,\n"
        << "                " << "(DATA_T**)&" + outputName + "["
                              << output_pos + "],\n"
        << "                " << identifier + "_poolingDesc\n"
        << "    " <<");\n";
}

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramOutputFunction(
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

void N2D2::CPP_cuDNN_PoolCellExport::generateCellProgramFree(Cell& cell,
    std::vector<std::string>& parentsName, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    for(int k = parentsName.size() - 1; k >= 0; --k) {

        prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << identifier
            << "_tensorDescIn[" << k << "]) );\n";

        prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << identifier
            << "_tensorDescOut[" << k << "]) );\n";

    }

    prog << "    CHECK_CUDNN_STATUS( cudnnDestroyPoolingDescriptor("
        << identifier << "_poolingDesc) );\n"
            "\n";
}
