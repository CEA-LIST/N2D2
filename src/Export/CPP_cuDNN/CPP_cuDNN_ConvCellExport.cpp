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

#include "Export/CPP_cuDNN/CPP_cuDNN_ConvCellExport.hpp"

N2D2::Registrar<N2D2::ConvCellExport>
N2D2::CPP_cuDNN_ConvCellExport::mRegistrar(
    "CPP_cuDNN", N2D2::CPP_cuDNN_ConvCellExport::generate);

N2D2::Registrar<N2D2::CPP_cuDNN_CellExport>
N2D2::CPP_cuDNN_ConvCellExport::mRegistrarType(
    ConvCell::Type, N2D2::CPP_cuDNN_ConvCellExport::getInstance);

void N2D2::CPP_cuDNN_ConvCellExport::generate(ConvCell& cell,
                                              const std::string& dirName)
{
    Utils::createDirectories(dirName + "/include");

    const std::string fileName = dirName + "/include/" + cell.getName()
                                 + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);
    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_cuDNN_CellExport::generateHeaderIncludes(cell, header);
    CPP_ConvCellExport::generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_cuDNN_ConvCellExport::generateHeaderFreeParameters(ConvCell
                                                                  & cell,
                                                                  std::ofstream
                                                                  & header)
{
    generateHeaderBias(cell, header);
    generateHeaderWeights(cell, header);
}

void N2D2::CPP_cuDNN_ConvCellExport::generateHeaderBias(ConvCell& cell,
                                                        std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_cuDNN_ConvCellExport::generateHeaderBiasVariable(ConvCell& cell,
                                                                std::ofstream
                                                                & header)
{
    header << "static WDATA_T " << cell.getName() << "_biases["
           << Utils::upperCase(cell.getName()) << "_NB_OUTPUTS] = ";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateHeaderBiasValues(ConvCell& cell,
                                                      std::ofstream& header)
{
    header << "{";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        if (cell.getParameter<bool>("NoBias"))
            header << "0";
        else
            CellExport::generateFreeParameter(
                cell, cell.getBias(output), header);
    }

    header << "};\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateHeaderWeights(ConvCell& cell,
                                                           std::ofstream
                                                           & header)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    // Flatten weights storage format
    header << "#define " << prefix << "_WEIGHTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS*" << prefix
           << "_KERNEL_WIDTH*" << prefix << "_KERNEL_HEIGHT)\n"
           << "static WDATA_T " << cell.getName() << "_weights_flatten["
           << prefix << "_WEIGHTS_SIZE] = {\n";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            for (unsigned int sy = 0; sy < cell.getKernelHeight(); ++sy) {
                for (unsigned int sx = 0; sx < cell.getKernelWidth(); ++sx) {
                    if (output > 0 || channel > 0 || sy > 0 || sx > 0)
                        header << ", ";

                    if (!cell.isConnection(channel, output))
                        header << "0";
                    else
                        CellExport::generateFreeParameter(
                            cell, cell.getWeight(output, channel, sx, sy),
                            header);
                }
            }
        }
    }

    header << "};\n\n";
}

std::unique_ptr<N2D2::CPP_cuDNN_ConvCellExport>
N2D2::CPP_cuDNN_ConvCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_cuDNN_ConvCellExport>(new CPP_cuDNN_ConvCellExport);
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramDesc(Cell& cell,
                                                             std::ofstream
                                                             & prog)
{
    generateCellProgramTensorDesc(cell, prog);
    generateCellProgramConvDesc(cell, prog);
    generateCellProgramFilterDesc(cell, prog);
}
void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramTensorDesc(Cell& cell,
                                                                   std::ofstream
                                                                   & prog)
{
    prog << "std::vector<cudnnTensorDescriptor_t> " << cell.getName()
         << "_tensorDescIn;\n"
         "cudnnTensorDescriptor_t " << cell.getName()
         << "_tensorDescOut;\n"
         "cudnnTensorDescriptor_t " << cell.getName()
         << "_biasesDesc;\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramConvDesc(Cell& cell,
                                                                 std::ofstream
                                                                 & prog)
{
    prog << "cudnnConvolutionDescriptor_t " << cell.getName() << "_convDesc;\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramFilterDesc(Cell& cell,
                                                                   std::ofstream
                                                                   & prog)
{
    prog << "std::vector<cudnnFilterDescriptor_t> " << cell.getName()
         << "_filterDesc;\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramActivationDesc(
    Cell& cell, std::ofstream& prog)
{
    prog << "cudnnActivationDescriptor_t " << cell.getName()
         << "_activationDesc;\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{
    prog << "size_t " << cell.getName() << "_sB = 0;\n"
            "void* " << cell.getName() << "_Workspace;\n"
            "std::vector<cudnnConvolutionFwdAlgo_t> "
            << cell.getName() << "_algo;\n"
            "std::vector<DATA_T *>" << cell.getName() << "_weights_cudnn;\n"
            "DATA_T *" << cell.getName() << "_bias_cudnn;\n"
            "\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellBuffer(const std::string
                                                        & bufferName,
                                                        std::ofstream& prog)
{
    prog << "std::vector<DATA_T *> " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramInitNetwork(
    Cell& cell,std::vector<std::string>& parentsName, std::ofstream& prog)
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
        if(prefixParent != "ENV") {
            prog << "    " << cell.getName() << "_chanHeightPerLayer.push_back("
                << prefixParent << "_OUTPUTS_HEIGHT);\n";
            prog << "    " << cell.getName() << "_chanWidthPerLayer.push_back("
                << prefixParent << "_OUTPUTS_WIDTH);\n";
        } else {
            prog << "    " << cell.getName() << "_chanHeightPerLayer.push_back("
                << prefixParent << "_SIZE_Y);\n";
            prog << "    " << cell.getName() << "_chanWidthPerLayer.push_back("
                << prefixParent << "_SIZE_X);\n";

        }

    }

    prog << "    setConvolution(batchSize,\n"
        << "                " << cell.getName() << "_nbChanPerLayer,\n"
        << "                " << cell.getName() << "_chanHeightPerLayer,\n"
        << "                " << cell.getName() << "_chanWidthPerLayer,\n"
        << "                " << prefix << "_PADDING_Y,\n"
        << "                " << prefix << "_PADDING_X,\n"
        << "                " << prefix << "_STRIDE_Y,\n"
        << "                " << prefix << "_STRIDE_X,\n"
        << "                " << prefix << "_SUB_SAMPLE_Y,\n"
        << "                " << prefix << "_SUB_SAMPLE_X,\n"
        << "                " << cell.getName() << "_weights_flatten,\n"
        << "                " << cell.getName() << "_weights_cudnn,\n"
        << "                " << cell.getName() << "_biases,\n"
        << "                " << cell.getName() << "_bias_cudnn,\n"
        << "                " << "context_handle,\n"
        << "                " << "context_tensorFormat,\n"
        << "                " << "context_dataType,\n"
        << "                " << cell.getName() << "_tensorDescIn,\n"
        << "                " << cell.getName() << "_tensorDescOut,\n"
        << "                " << prefix << "_ACTIVATION,\n"
        << "                " << cell.getName() << "_algo,\n"
        << "                " << cell.getName() << "_sB,\n"
        << "                " << "&" << cell.getName() << "_Workspace,\n"
        << "                " << prefix << "_NB_OUTPUTS,\n"
        << "                " << prefix << "_OUTPUTS_HEIGHT,\n"
        << "                " << prefix << "_OUTPUTS_WIDTH,\n"
        << "                " << prefix << "_KERNEL_HEIGHT,\n"
        << "                " << prefix << "_KERNEL_WIDTH,\n"
        << "                " << cell.getName() <<  "_biasesDesc,\n"
        << "                " << cell.getName() << "_filterDesc,\n"
        << "                " << cell.getName() << "_convDesc);\n\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramInitBuffer(
    Cell& cell, const std::string& bufferName, std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << bufferName << "buffer[" << prefix
        << "_OUTPUT_OFFSET], sizeof(DATA_T)*"
        << Utils::upperCase(cell.getName())
        << "_OUTPUTS_SIZE*batchSize));\n\n\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& output_pos,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string prefix = Utils::upperCase(cell.getName());
    const std::string proto = (funcProto.empty()) ? "    convcell" : funcProto;

    prog << proto
        << "(\n"
        << "                " << "context_handle,\n"
        << "                " << prefix + "_ACTIVATION,\n"
        << "                " << cell.getName() + "_algo,\n"
        << "                " << cell.getName() + "_Workspace,\n"
        << "                " << cell.getName() + "_sB,\n"
        << "                " << cell.getName() + "_tensorDescIn,\n"
        << "                " << inputName + ",\n"
        << "                " << prefix + "_NO_BIAS,\n"
        << "                " << cell.getName() + "_tensorDescOut,\n"
        << "                " << "(DATA_T**)&" + outputName
                              << "[" + output_pos + "],\n"
        << "                " << cell.getName() + "_biases" + "Desc,\n"
        << "                " << cell.getName() + "_bias_cudnn,\n"
        << "                " << cell.getName() + "_filterDesc,\n"
        << "                " << cell.getName() + "_convDesc,\n"
        << "                " << cell.getName() + "_weights_cudnn\n"
        << "        " << ");\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramOutputFunction(
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

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramFree(Cell& cell,
    std::vector<std::string>& parentsName, std::ofstream& prog)
{
    for(int k = parentsName.size() - 1; k >= 0; --k) {

        prog << "    CHECK_CUDA_STATUS( cudaFree(" << cell.getName()
            << "_weights_cudnn[" << k << "]) );\n";

        prog << "    CHECK_CUDNN_STATUS( cudnnDestroyFilterDescriptor("
            << cell.getName()
            << "_filterDesc[" << k << "]) );\n";

        prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << cell.getName()
            << "_tensorDescIn[" << k << "]) );\n";
    }
    prog << "    CHECK_CUDA_STATUS( cudaFree(" << cell.getName()
        << "_bias_cudnn) );\n";

    prog << "    CHECK_CUDNN_STATUS( cudnnDestroyConvolutionDescriptor("
            << cell.getName() << "_convDesc) );\n"
            "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << cell.getName() << "_biasesDesc) );\n"
            "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << cell.getName() << "_tensorDescOut) );\n"
            //"#if CUDNN_VERSION >= 5000\n  CHECK_CUDNN_STATUS( cudnnDestroyActivationDescriptor("<< cell.getName() << "_activationDesc) );\n#endif\n"
            "\n";

}
