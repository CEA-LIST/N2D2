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
    C_CellExport::generateHeaderBegin(cell, header);
    CPP_cuDNN_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    C_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_cuDNN_ConvCellExport::generateHeaderConstants(ConvCell& cell,
                                                             std::ofstream
                                                             & header)
{
    C_ConvCellExport::generateHeaderConstants(cell, header);

    const std::string prefix = Utils::upperCase(cell.getName());

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_OUTPUTS_WIDTH*" << prefix
           << "_OUTPUTS_HEIGHT)\n"
              "#define " << prefix << "_CHANNELS_SIZE (" << prefix
           << "_NB_CHANNELS*" << prefix << "_CHANNELS_WIDTH*" << prefix
           << "_CHANNELS_HEIGHT)\n"
              "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix
           << "_OUTPUTS_SIZE, " << prefix << "_CHANNELS_SIZE))\n"
                                             "#define " << prefix << "_NO_BIAS "
           << (cell.getParameter<bool>("NoBias") ? "1" : "0") << "\n";
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
    C_ConvCellExport::generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_cuDNN_ConvCellExport::generateHeaderBiasVariable(ConvCell& cell,
                                                                std::ofstream
                                                                & header)
{
    header << "static WDATA_T " << cell.getName() << "_biases["
           << Utils::upperCase(cell.getName()) << "_NB_OUTPUTS] = ";
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
            for (unsigned int sy = cell.getKernelHeight(); sy > 0; --sy) {
                for (unsigned int sx = cell.getKernelWidth(); sx > 0; --sx) {
                    if (output > 0 || channel > 0 || sy < cell.getKernelHeight()
                        || sx < cell.getKernelWidth())
                        header << ", ";

                    if (!cell.isConnection(channel, output))
                        header << "0";
                    else
                        CellExport::generateFreeParameter(
                            cell,
                            cell.getWeight(output, channel, sx - 1, sy - 1),
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
    prog << "cudnnTensorDescriptor_t " << cell.getName()
         << "_tensorDescIn;\n"
            "cudnnTensorDescriptor_t " << cell.getName()
         << "_tensorDescOut;\n"
            "cudnnTensorDescriptor_t " << cell.getName() << "_biasesDesc;\n";
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
    prog << "cudnnFilterDescriptor_t " << cell.getName() << "_filterDesc;\n";
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
    prog << "size_t " << cell.getName() << "_sB;\n"
                                           "void* " << cell.getName()
         << "_Workspace;\n"
            "cudnnConvolutionFwdAlgo_t " << cell.getName() << "_algo;\n"
                                                              "DATA_T *"
         << cell.getName() << "_weights_cudnn(NULL);\n"
                              "DATA_T *" << cell.getName()
         << "_bias_cudnn(NULL);\n"
            "\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellBuffer(const std::string
                                                        & bufferName,
                                                        std::ofstream& prog)
{
    prog << "DATA_T * " << bufferName << ";\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramInitNetwork(
    Cell& cell, std::ofstream& prog)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    prog << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_tensorDescIn);\n"
         << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_tensorDescOut);\n"
         << " cudnnCreateTensorDescriptor(&" << cell.getName()
         << "_biasesDesc);\n";

    prog << " cudnnCreateFilterDescriptor(&" << cell.getName()
         << "_filterDesc);\n"
         << " cudnnCreateConvolutionDescriptor(&" << cell.getName()
         << "_convDesc);\n";

    prog << " setConvolution(batchSize, " << prefix << "_NB_CHANNELS, "
         << prefix << "_CHANNELS_HEIGHT, " << prefix << "_CHANNELS_WIDTH, "
         << prefix << "_PADDING_Y, " << prefix << "_PADDING_X, " << prefix
         << "_STRIDE_Y, " << prefix << "_STRIDE_X, " << prefix
         << "_SUB_SAMPLE_Y, " << prefix
         << "_SUB_SAMPLE_X, context_handle, context_tensorFormat, "
            "context_dataType," << cell.getName() << "_tensorDescIn, "
         << cell.getName() << "_tensorDescOut, " << prefix << "_ACTIVATION, "
         << cell.getName() << "_algo, " << cell.getName() << "_sB, &"
         << cell.getName() << "_Workspace, " << prefix << "_NB_OUTPUTS, "
         << prefix << "_OUTPUT_OFFSET*batchSize, " << prefix
         << "_KERNEL_HEIGHT, " << prefix << "_KERNEL_WIDTH, " << cell.getName()
         << "_biasesDesc, " << cell.getName() << "_filterDesc, "
         << cell.getName() << "_convDesc);\n\n";

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

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramInitBuffer(
    const std::string& bufferName, std::ofstream& prog)
{
    prog << " CHECK_CUDA_STATUS(cudaMalloc(&" << bufferName
         << "buffer, sizeof(DATA_T)*" << Utils::upperCase(bufferName)
         << "OUTPUTS_SIZE*batchSize));\n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramFunction(
    Cell& cell,
    const std::string& inputName,
    const std::string& outputName,
    std::ofstream& prog,
    const std::string& funcProto)
{
    const std::string prefix = Utils::upperCase(cell.getName());
    const std::string proto = (funcProto.empty()) ? " convcell" : funcProto;

    prog << proto << "( "
         << "context_handle, " << prefix + "_ACTIVATION, "
         << cell.getName() + "_algo, &" << cell.getName() + "_Workspace, "
         << cell.getName() + "_sB, "
         << cell.getName() + "_tensorDescIn, " + inputName + ", "
         << prefix + "_OUTPUT_OFFSET*batchSize, " << prefix + "_NO_BIAS, "
         << cell.getName() + "_tensorDescOut, (DATA_T**)&" + outputName + ", "
         << cell.getName() + "_biases" + "Desc, "
         << cell.getName() + "_bias_cudnn, " << cell.getName() + "_filterDesc, "
         << cell.getName() + "_convDesc, "
         << cell.getName() + "_weights_cudnn); \n";
}

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramOutputFunction(
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

void N2D2::CPP_cuDNN_ConvCellExport::generateCellProgramFree(Cell& cell,
                                                             std::ofstream
                                                             & prog)
{
    prog << " CHECK_CUDA_STATUS( cudaFree(" << cell.getName()
         << "_weights_cudnn) );\n"
            " CHECK_CUDA_STATUS( cudaFree(" << cell.getName()
         << "_bias_cudnn) );\n";

    prog << " CHECK_CUDNN_STATUS( cudnnDestroyConvolutionDescriptor("
         << cell.getName()
         << "_convDesc) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyFilterDescriptor("
         << cell.getName()
         << "_filterDesc) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName()
         << "_biasesDesc) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName()
         << "_tensorDescIn) );\n"
            " CHECK_CUDNN_STATUS( cudnnDestroyTensorDescriptor("
         << cell.getName() << "_tensorDescOut) );\n"
        //"#if CUDNN_VERSION >= 5000\n  CHECK_CUDNN_STATUS(
        // cudnnDestroyActivationDescriptor("<< cell.getName() <<
        //"_activationDesc) );\n#endif\n"
                              "\n";
}
