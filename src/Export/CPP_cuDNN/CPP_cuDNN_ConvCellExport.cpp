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
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

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
    const std::string identifier = Utils::CIdentifier(cell.getName());

    header << "static WDATA_T " << identifier << "_biases["
           << Utils::upperCase(identifier) << "_NB_OUTPUTS] = ";
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
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    // Flatten weights storage format
    header << "#define " << prefix << "_WEIGHTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS*" << prefix
           << "_KERNEL_WIDTH*" << prefix << "_KERNEL_HEIGHT)\n"
           << "static WDATA_T " << identifier << "_weights_flatten["
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
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "std::vector<cudnnTensorDescriptor_t> " << identifier
         << "_tensorDescIn;\n"
         "cudnnTensorDescriptor_t " << identifier
         << "_tensorDescOut;\n"
         "cudnnTensorDescriptor_t " << identifier
         << "_biasesDesc;\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramConvDesc(Cell& cell,
                                                                 std::ofstream
                                                                 & prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "cudnnConvolutionDescriptor_t " << identifier << "_convDesc;\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramFilterDesc(Cell& cell,
                                                                   std::ofstream
                                                                   & prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "std::vector<cudnnFilterDescriptor_t> " << identifier
         << "_filterDesc;\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramActivationDesc(
    Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "cudnnActivationDescriptor_t " << identifier
         << "_activationDesc;\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "size_t " << identifier << "_sB = 0;\n"
            "void* " << identifier << "_Workspace;\n"
            "std::vector<cudnnConvolutionFwdAlgo_t> "
            << identifier << "_algo;\n"
            "std::vector<DATA_T *>" << identifier << "_weights_cudnn;\n"
            "DATA_T *" << identifier << "_bias_cudnn;\n"
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
        if(prefixParent != "ENV") {
            prog << "    " << identifier << "_chanHeightPerLayer.push_back("
                << prefixParent << "_OUTPUTS_HEIGHT);\n";
            prog << "    " << identifier << "_chanWidthPerLayer.push_back("
                << prefixParent << "_OUTPUTS_WIDTH);\n";
        } else {
            prog << "    " << identifier << "_chanHeightPerLayer.push_back("
                << prefixParent << "_SIZE_Y);\n";
            prog << "    " << identifier << "_chanWidthPerLayer.push_back("
                << prefixParent << "_SIZE_X);\n";

        }

    }

    prog << "    setConvolution(batchSize,\n"
        << "                " << identifier << "_nbChanPerLayer,\n"
        << "                " << identifier << "_chanHeightPerLayer,\n"
        << "                " << identifier << "_chanWidthPerLayer,\n"
        << "                " << prefix << "_PADDING_Y,\n"
        << "                " << prefix << "_PADDING_X,\n"
        << "                " << prefix << "_STRIDE_Y,\n"
        << "                " << prefix << "_STRIDE_X,\n"
        << "                " << prefix << "_SUB_SAMPLE_Y,\n"
        << "                " << prefix << "_SUB_SAMPLE_X,\n"
        << "                " << identifier << "_weights_flatten,\n"
        << "                " << identifier << "_weights_cudnn,\n"
        << "                " << identifier << "_biases,\n"
        << "                " << identifier << "_bias_cudnn,\n"
        << "                " << "CudaContext::cudnnHandle(),\n"
        << "                " << "context_tensorFormat,\n"
        << "                " << "context_dataType,\n"
        << "                " << identifier << "_tensorDescIn,\n"
        << "                " << identifier << "_tensorDescOut,\n"
        << "                " << prefix << "_ACTIVATION,\n"
        << "                " << identifier << "_algo,\n"
        << "                " << identifier << "_sB,\n"
        << "                " << "&" << identifier << "_Workspace,\n"
        << "                " << prefix << "_NB_OUTPUTS,\n"
        << "                " << prefix << "_OUTPUTS_HEIGHT,\n"
        << "                " << prefix << "_OUTPUTS_WIDTH,\n"
        << "                " << prefix << "_KERNEL_HEIGHT,\n"
        << "                " << prefix << "_KERNEL_WIDTH,\n"
        << "                " << identifier <<  "_biasesDesc,\n"
        << "                " << identifier << "_filterDesc,\n"
        << "                " << identifier << "_convDesc);\n\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramInitBuffer(
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

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& output_pos,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto = (funcProto.empty()) ? "    convcell" : funcProto;

    prog << proto
        << "(\n"
        << "                " << "CudaContext::cudnnHandle(),\n"
        << "                " << prefix + "_ACTIVATION,\n"
        << "                " << identifier + "_algo,\n"
        << "                " << identifier + "_Workspace,\n"
        << "                " << identifier + "_sB,\n"
        << "                " << identifier + "_tensorDescIn,\n"
        << "                " << inputName + ",\n"
        << "                " << prefix + "_NO_BIAS,\n"
        << "                " << identifier + "_tensorDescOut,\n"
        << "                " << "(DATA_T**)&" + outputName
                              << "[" + output_pos + "],\n"
        << "                " << identifier + "_biases" + "Desc,\n"
        << "                " << identifier + "_bias_cudnn,\n"
        << "                " << identifier + "_filterDesc,\n"
        << "                " << identifier + "_convDesc,\n"
        << "                " << identifier + "_weights_cudnn\n"
        << "        " << ");\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramOutputFunction(
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

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramFree(Cell& cell,
    std::vector<std::string>& parentsName, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    for(int k = parentsName.size() - 1; k >= 0; --k) {

        prog << "    CHECK_CUDA_STATUS( cudaFree(" << identifier
            << "_weights_cudnn[" << k << "]) );\n";

        prog << "    CHECK_CUDNN_STATUS( cudnnDestroyFilterDescriptor("
            << identifier
            << "_filterDesc[" << k << "]) );\n";

        prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << identifier
            << "_tensorDescIn[" << k << "]) );\n";
    }
    prog << "    CHECK_CUDA_STATUS( cudaFree(" << identifier
        << "_bias_cudnn) );\n";

    prog << "    CHECK_CUDNN_STATUS( cudnnDestroyConvolutionDescriptor("
            << identifier << "_convDesc) );\n"
            "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << identifier << "_biasesDesc) );\n"
            "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << identifier << "_tensorDescOut) );\n"
            //"#if CUDNN_VERSION >= 5000\n  CHECK_CUDNN_STATUS( cudnnDestroyActivationDescriptor("<< identifier << "_activationDesc) );\n#endif\n"
            "\n";

}
