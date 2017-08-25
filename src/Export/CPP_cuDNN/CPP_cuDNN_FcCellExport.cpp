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

#include "Export/CPP_cuDNN/CPP_cuDNN_FcCellExport.hpp"

N2D2::Registrar<N2D2::FcCellExport> N2D2::CPP_cuDNN_FcCellExport::mRegistrar(
    "CPP_cuDNN", N2D2::CPP_cuDNN_FcCellExport::generate);

N2D2::Registrar<N2D2::CPP_cuDNN_CellExport>
N2D2::CPP_cuDNN_FcCellExport::mRegistrarType(
    FcCell::Type, N2D2::CPP_cuDNN_FcCellExport::getInstance);

void N2D2::CPP_cuDNN_FcCellExport::generate(FcCell& cell,
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
    generateHeaderFreeParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderConstants(FcCell& cell,
                                                           std::ofstream
                                                           & header)
{
    // Constants
    const unsigned int channelsSize = cell.getNbChannels()
                                      * cell.getChannelsWidth()
                                      * cell.getChannelsHeight();
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs()
           << "\n"
              "#define " << prefix << "_NB_CHANNELS " << channelsSize << "\n\n";

    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);

    if (cellFrame != NULL) {
        header << "#define " << prefix << "_ACTIVATION "
               << ((cellFrame->getActivation())
                       ? cellFrame->getActivation()->getType()
                       : "Linear") << "\n";
    }

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix
           << "_NB_OUTPUTS)\n"
              "#define " << prefix << "_CHANNELS_SIZE (" << prefix
           << "_NB_CHANNELS)\n"
              "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix
           << "_OUTPUTS_SIZE, " << prefix << "_CHANNELS_SIZE))\n"
              "#define " << prefix
           << "_CHANNELS_HEIGHT 1\n"
              "#define " << prefix << "_OUTPUTS_HEIGHT 1\n"
              "#define " << prefix << "_OUTPUTS_WIDTH 1\n"
              "#define " << prefix << "_CHANNELS_WIDTH 1\n"
                                      "#define " << prefix << "_NO_BIAS "
           << (cell.getParameter<bool>("NoBias") ? "1" : "0") << "\n";
}


void N2D2::CPP_cuDNN_FcCellExport::generateHeaderFreeParameters(FcCell& cell,
                                                                std::ofstream
                                                                & header)
{
    generateHeaderBias(cell, header);

    if (mThreshold > 0.0)
        generateHeaderWeightsSparse(cell, header);
    else
        generateHeaderWeights(cell, header);
}

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderBias(FcCell& cell,
                                                      std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderBiasVariable(FcCell& cell,
                                                              std::ofstream
                                                              & header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    header << "static WDATA_T " << identifier << "_biases["
           << Utils::upperCase(identifier) << "_NB_OUTPUTS] = ";
}

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderBiasValues(FcCell& cell,
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

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderWeights(FcCell& cell,
                                                         std::ofstream& header)
{
    const unsigned int channelsSize = cell.getNbChannels()
                                      * cell.getChannelsWidth()
                                      * cell.getChannelsHeight();
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_NB_WEIGHTS (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n\n";

    // Weights flatten
    header << "#define " << prefix << "_WEIGHTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n"
           << "static WDATA_T " << identifier << "_weights_flatten["
           << prefix << "_WEIGHTS_SIZE] = {\n";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < channelsSize; ++channel) {
            if (output > 0 || channel > 0)
                header << ", ";

            CellExport::generateFreeParameter(
                cell, cell.getWeight(output, channel), header);
        }
    }

    header << "};\n\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderWeightsSparse(FcCell& cell,
                                                       std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const unsigned int channelsSize = cell.getNbChannels()
                                      * cell.getChannelsWidth()
                                      * cell.getChannelsHeight();

    std::vector<double> weights;
    std::vector<unsigned int> offsets;
    unsigned int offset = 0;

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < channelsSize; ++channel) {
            double w = cell.getWeight(output, channel);

            if (std::fabs(w) >= mThreshold) {
                weights.push_back(w);
                offsets.push_back(offset);
                offset = 1;
            } else
                ++offset;
        }
    }

    const unsigned int nbWeights = weights.size();

    header << "#define " << prefix << "_NB_WEIGHTS " << nbWeights << "\n"
           << "static WDATA_T " << identifier << "_weights_sparse["
           << prefix << "_NB_WEIGHTS] = {\n";

    for (unsigned int i = 0; i < nbWeights; ++i) {
        if (i > 0)
            header << ", ";

        CellExport::generateFreeParameter(cell, weights[i], header);
    }

    header << "};\n\n";

    header << "static unsigned short " << identifier << "_weights_offsets["
           << prefix << "_NB_WEIGHTS] = {\n";

    for (unsigned int i = 0; i < nbWeights; ++i) {
        if (i > 0)
            header << ", ";

        header << offsets[i];
    }

    header << "};\n\n";

    std::cout << Utils::cnotice << "Sparse weights ratio: " << nbWeights << "/"
              << (cell.getNbOutputs() * channelsSize) << " ("
              << 100.0
                 * (nbWeights / (double)(cell.getNbOutputs() * channelsSize))
              << "%)" << Utils::cdef << std::endl;
}

std::unique_ptr<N2D2::CPP_cuDNN_FcCellExport>
N2D2::CPP_cuDNN_FcCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_cuDNN_FcCellExport>(new CPP_cuDNN_FcCellExport);
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramDesc(Cell& cell,
                                                           std::ofstream& prog)
{

    generateCellProgramTensorDesc(cell, prog);
    // generateCellProgramActivationDesc(cell, prog);
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramTensorDesc(Cell& cell,
                                                                 std::ofstream
                                                                 & prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "std::vector<cudnnTensorDescriptor_t> "
        << identifier << "_tensorDescIn;\n"
        "cudnnTensorDescriptor_t " << identifier
        << "_tensorDescOut;\n"
        "cudnnTensorDescriptor_t " << identifier << "_biasesDesc;\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramActivationDesc(
    Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "cudnnActivationDescriptor_t " << identifier
         << "_activationDesc;\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "std::vector<DATA_T *>" << identifier << "_weights_cudnn;\n"
            "DATA_T *" << identifier << "_bias_cudnn;\n"
            "\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellBuffer(const std::string
                                                      & bufferName,
                                                      std::ofstream& prog)
{
    prog << "std::vector<DATA_T *> " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramInitNetwork(
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

    prog << "    setFc(batchSize,\n"
        << "                " << identifier << "_nbChanPerLayer,\n"
        << "                " << identifier << "_chanHeightPerLayer,\n"
        << "                " << identifier << "_chanWidthPerLayer,\n"
        << "                " << identifier << "_tensorDescIn,\n"
        << "                " << identifier << "_weights_flatten,\n"
        << "                " << identifier << "_weights_cudnn,\n"
        << "                " << identifier << "_biases,\n"
        << "                " << identifier << "_bias_cudnn,\n"
        << "                " << "context_tensorFormat,\n"
        << "                " << "context_dataType,\n"
        << "                " << identifier << "_tensorDescOut,\n"
        << "                " << prefix << "_ACTIVATION,\n"
        << "                " << prefix << "_NB_OUTPUTS);\n\n\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramInitBuffer(Cell& cell,
    const std::string& bufferName, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "    CHECK_CUDA_STATUS(cudaMalloc(&"
        << bufferName << "buffer[" << prefix
        << "_OUTPUT_OFFSET], sizeof(DATA_T)*"
        << prefix
        << "_OUTPUTS_SIZE*batchSize));\n\n\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& output_pos,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto = (funcProto.empty()) ?
        "    fullyConnected" : funcProto;

    prog << proto
        << "(\n"
        << "                " << prefix + "_NB_CHANNELS,\n"
        << "                " << "CudaContext::cudnnHandle(),\n"
        << "                " << "CudaContext::cublasHandle(),\n"
        << "                " << prefix + "_ACTIVATION,\n"
        << "                " << identifier + "_tensorDescIn,\n"
        << "                " << inputName + ",\n"
        << "                " << identifier + "_tensorDescOut,\n"
        << "                " << prefix + "_NB_OUTPUTS,\n"
        << "                " << prefix + "_NO_BIAS,\n"
        << "                " << "(DATA_T**)&" + outputName + "["
                              << output_pos + "],\n"
        << "                " << identifier + "_bias_cudnn,\n"
        << "                " << "ones_vector_buffer,\n"
        << "                " << identifier + "_weights_cudnn\n"
        << "    " <<");\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramOutputFunction(
    Cell& cell,
    const std::string& outputDataName,
    const std::string& outputName,
    std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "    output_generation(batchSize, "
            << prefix << "_NB_OUTPUTS, "
            << outputDataName << ", "
            << outputName << ");\n";
}
void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramFree(
    Cell& cell, std::vector<std::string>& parentsName, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

   for(int k = parentsName.size() - 1; k >= 0; --k) {

        prog << "    CHECK_CUDA_STATUS( cudaFree(" << identifier
            << "_weights_cudnn[" << k << "]) );\n";

        prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
            << identifier << "_tensorDescIn[" << k << "]) );\n";
    }
    prog << "    CHECK_CUDA_STATUS( cudaFree(" << identifier
        << "_bias_cudnn) );\n";

    prog << "    CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
        << identifier << "_tensorDescOut) );\n"
            //"#if CUDNN_VERSION >= 5000\n  CHECK_CUDNN_STATUS( cudnnDestroyActivationDescriptor("<< identifier << "_activationDesc) );\n#endif\n"
            << "\n";
}
