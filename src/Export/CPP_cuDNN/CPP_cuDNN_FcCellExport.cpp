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
    Utils::createDirectories(dirName + "/include");

    const std::string fileName = dirName + "/include/" + cell.getName()
                                 + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    C_CellExport::generateHeaderBegin(cell, header);
    CPP_cuDNN_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    C_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderConstants(FcCell& cell,
                                                           std::ofstream
                                                           & header)
{
    C_FcCellExport::generateHeaderConstants(cell, header);

    const std::string prefix = Utils::upperCase(cell.getName());

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix
           << "_NB_OUTPUTS)\n"
              "#define " << prefix << "_CHANNELS_SIZE (" << prefix
           << "_NB_CHANNELS)\n"
              "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix
           << "_OUTPUTS_SIZE, " << prefix << "_CHANNELS_SIZE))\n"
                                             "#define " << prefix
           << "_CHANNELS_HEIGHT 1\n"
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
        C_FcCellExport::generateHeaderWeightsSparse(cell, header);
    else
        generateHeaderWeights(cell, header);
}

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderBias(FcCell& cell,
                                                      std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    C_FcCellExport::generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderBiasVariable(FcCell& cell,
                                                              std::ofstream
                                                              & header)
{
    header << "static WDATA_T " << cell.getName() << "_biases["
           << Utils::upperCase(cell.getName()) << "_NB_OUTPUTS] = ";
}

void N2D2::CPP_cuDNN_FcCellExport::generateHeaderWeights(FcCell& cell,
                                                         std::ofstream& header)
{
    const unsigned int channelsSize = cell.getNbChannels()
                                      * cell.getChannelsWidth()
                                      * cell.getChannelsHeight();
    const std::string prefix = Utils::upperCase(cell.getName());

    header << "#define " << prefix << "_NB_WEIGHTS (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n\n";

    // Weights flatten
    header << "#define " << prefix << "_WEIGHTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n"
           << "static WDATA_T " << cell.getName() << "_weights_flatten["
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
    prog << "cudnnTensorDescriptor_t " << cell.getName()
         << "_tensorDescIn;\n"
            "cudnnTensorDescriptor_t " << cell.getName()
         << "_tensorDescOut;\n"
            "cudnnTensorDescriptor_t " << cell.getName() << "_biasesDesc;\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramActivationDesc(
    Cell& cell, std::ofstream& prog)
{
    prog << "cudnnActivationDescriptor_t " << cell.getName()
         << "_activationDesc;\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramGlobalDefinition(
    Cell& cell, std::ofstream& prog)
{

    prog << "DATA_T *" << cell.getName() << "_weights_cudnn(NULL);\n"
                                            "DATA_T *" << cell.getName()
         << "_bias_cudnn(NULL);\n"
            "\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellBuffer(const std::string
                                                      & bufferName,
                                                      std::ofstream& prog)
{
    prog << "DATA_T * " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramInitNetwork(Cell& cell,
                                                                  std::ofstream
                                                                  & prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_tensorDescIn);\n"
         << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_tensorDescOut);\n"
         << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_biasesDesc);\n";

    prog << " setFc(batchSize, " << prefix << "_NB_CHANNELS, " << prefix
         << "_CHANNELS_HEIGHT, " << prefix
         << "_CHANNELS_WIDTH, "
            "context_handle, context_tensorFormat, context_dataType,"
         << cell.getName() << "_tensorDescIn, " << cell.getName()
         << "_tensorDescOut, " << prefix << "_ACTIVATION, " << prefix
         << "_NB_OUTPUTS, " << cell.getName() << "_biasesDesc);\n\n";
    prog << " CHECK_CUDA_STATUS( cudaMalloc(&" << cell.getName()
         << "_weights_cudnn, " << prefix << "_WEIGHTS_SIZE*sizeof(DATA_T)) );\n"
         << " CHECK_CUDA_STATUS( cudaMemcpy(" << cell.getName()
         << "_weights_cudnn, " << cell.getName() << "_weights_flatten, "
         << prefix
         << "_WEIGHTS_SIZE*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n";

    prog << " CHECK_CUDA_STATUS( cudaMalloc(&" << cell.getName()
         << "_bias_cudnn, " << prefix << "_NB_OUTPUTS*sizeof(DATA_T)) );\n"
         << " CHECK_CUDA_STATUS( cudaMemcpy(" << cell.getName()
         << "_bias_cudnn, " << cell.getName() << "_biases, " << prefix
         << "_NB_OUTPUTS*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n"
            "\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramInitBuffer(
    const std::string& bufferName, std::ofstream& prog)
{
    prog << " CHECK_CUDA_STATUS(cudaMalloc(&" << bufferName
         << "buffer, sizeof(DATA_T)*" << Utils::upperCase(bufferName)
         << "OUTPUTS_SIZE*batchSize));\n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string prefix = Utils::upperCase(cell.getName());
    const std::string proto = (funcProto.empty()) ? " fullyConnected"
                                                  : funcProto;

    prog << proto << "(batchSize, "
         << prefix + "_NB_CHANNELS, context_handle, context_cublasHandle, "
         << prefix + "_ACTIVATION, "
         << cell.getName() + "_tensorDescIn, " + inputName + ", "
         << cell.getName() + "_tensorDescOut, " << prefix + "_NB_OUTPUTS, "
         << prefix + "_OUTPUT_OFFSET*batchSize," + prefix
            + "_NO_BIAS, (DATA_T**)&" + outputName + ", "
         << cell.getName() + "_biasesDesc, " << cell.getName() + "_bias_cudnn,"
         << "ones_vector_buffer, " << cell.getName() + "_weights_cudnn); \n";
}

void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramOutputFunction(
    Cell& cell,
    const std::string& outputDataName,
    const std::string& outputName,
    std::ofstream& prog)
{
    prog << " output_generation(batchSize, " << Utils::upperCase(cell.getName())
         << "_NB_OUTPUTS, " << outputDataName << ", " << outputName << ");\n";
}
void N2D2::CPP_cuDNN_FcCellExport::generateCellProgramFree(Cell& cell,
                                                           std::ofstream& prog)
{
    prog << " CHECK_CUDA_STATUS( cudaFree(" << cell.getName()
         << "_weights_cudnn) );\n"
            " CHECK_CUDA_STATUS( cudaFree(" << cell.getName()
         << "_bias_cudnn) );\n";

    prog << " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName()
         << "_biasesDesc) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName()
         << "_tensorDescIn) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName() << "_tensorDescOut) );\n"
        //" CHECK_CUDNN_STATUS( cudnnDestroyActivationDescriptor("<<
        // cell.getName() << "_activationDesc) );\n"
                              "\n";
}
