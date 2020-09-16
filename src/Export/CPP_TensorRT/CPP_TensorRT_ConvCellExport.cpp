/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Export/CPP_TensorRT/CPP_TensorRT_ConvCellExport.hpp"

N2D2::Registrar<N2D2::ConvCellExport>
N2D2::CPP_TensorRT_ConvCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_ConvCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_ConvCellExport::mRegistrarType(
    ConvCell::Type, N2D2::CPP_TensorRT_ConvCellExport::getInstance);

void N2D2::CPP_TensorRT_ConvCellExport::generate(ConvCell& cell,
                                              const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn/include");
    Utils::createDirectories(dirName + "/dnn/weights");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    const std::string weightName = dirName + "/dnn//weights/"
        + Utils::CIdentifier(cell.getName()) + "_weights.syntxt";
    const std::string biasName = dirName + "/dnn//weights/"
        + Utils::CIdentifier(cell.getName()) + "_bias.syntxt";

    std::ofstream weights(weightName.c_str());
    std::ofstream bias(biasName.c_str());
    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);
    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_TensorRT_CellExport::generateHeaderIncludes(cell, header);
    CPP_ConvCellExport::generateHeaderConstants(cell, header);
    CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);


    generateFileBias(cell, bias);
    bias.close();

    generateFileWeights(cell, weights);
    weights.close();
}

void N2D2::CPP_TensorRT_ConvCellExport::generateHeaderFreeParameters(ConvCell
                                                                  & cell,
                                                                  std::ofstream
                                                                  & header)
{
    //generateHeaderBias(cell, header);
    generateHeaderWeights(cell, header);
}

void N2D2::CPP_TensorRT_ConvCellExport::generateHeaderBias(ConvCell& cell,
                                                        std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_TensorRT_ConvCellExport::generateHeaderBiasVariable(ConvCell& cell,
                                                                std::ofstream
                                                                & header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    header << "static BDATA_T " << identifier << "_biases["
           << Utils::upperCase(identifier) << "_NB_OUTPUTS] = ";
}

void N2D2::CPP_TensorRT_ConvCellExport::generateHeaderBiasValues(ConvCell& cell,
                                                      std::ofstream& header)
{
    header << "{";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        if (cell.getParameter<bool>("NoBias"))
            header << "0";
        else {
            Tensor<Float_T> bias;
            cell.getBias(output, bias);

            CellExport::generateFreeParameter( bias(0), header);
        }
    }

    header << "};\n";
}

void N2D2::CPP_TensorRT_ConvCellExport::generateHeaderWeights(ConvCell& cell,
                                                           std::ofstream
                                                           & header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    // Flatten weights storage format
    header << "#define " << prefix << "_WEIGHTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS*" << prefix
           << "_KERNEL_WIDTH*" << prefix << "_KERNEL_HEIGHT)\n";
           //<< "static WDATA_T " << identifier << "_weights_flatten["
           //<< prefix << "_WEIGHTS_SIZE] = {\n";
/*
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
                        CellExport::generateFreeParameter(cell.getWeight(output, channel, sx, sy),
                                                          header);
                }
            }
        }
    }

    header << "};\n\n";
    */
}

void N2D2::CPP_TensorRT_ConvCellExport::generateFileBias(ConvCell& cell,
                                                      std::ofstream& file)
{

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            file << " ";

        if (cell.getParameter<bool>("NoBias"))
            file << "0";
        else {
            Tensor<Float_T> bias;
            cell.getBias(output, bias);

            CellExport::generateFreeParameter(bias(0), file, false);
        }
    }

}

void N2D2::CPP_TensorRT_ConvCellExport::generateFileWeights(ConvCell& cell,
                                                           std::ofstream& file)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel)
        {
            Tensor<Float_T> kernel;
            cell.getWeight(output, channel, kernel);

            for (unsigned int sy = 0; sy < cell.getKernelHeight(); ++sy) {
                for (unsigned int sx = 0; sx < cell.getKernelWidth(); ++sx) {
                    if (output > 0 || channel > 0 || sy > 0 || sx > 0)
                        file << " ";

                    if (!cell.isConnection(channel, output))
                        file << "0";
                    else
                        CellExport::generateFreeParameter(kernel(sx, sy), file, false);
                }
            }
        }
    }

}

std::unique_ptr<N2D2::CPP_TensorRT_ConvCellExport>
N2D2::CPP_TensorRT_ConvCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_TensorRT_ConvCellExport>(new CPP_TensorRT_ConvCellExport);
}

void N2D2::CPP_TensorRT_ConvCellExport
    ::generateCellProgramDescriptors(Cell& cell, std::ofstream& prog)
{
    generateConvProgramTensorDesc(cell, prog);
    generateConvProgramLayerDesc(cell, prog);
    generateConvProgramFilterDesc(cell, prog);
    generateConvProgramActivationDesc(cell, prog);
}

void N2D2::CPP_TensorRT_ConvCellExport
    ::generateConvProgramTensorDesc(Cell& cell, std::ofstream& prog)
{

    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::ITensor* " << identifier << "_tensor;\n";
}

void N2D2::CPP_TensorRT_ConvCellExport
    ::generateConvProgramLayerDesc(Cell& cell,std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::IConvolutionLayer* " << identifier << "_layer;\n";
}

void N2D2::CPP_TensorRT_ConvCellExport
    ::generateConvProgramFilterDesc(Cell& cell,std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::Weights " << identifier
         << "_filter;\n";
    prog << "nvinfer1::Weights " << identifier
         << "_bias;\n";
}

void N2D2::CPP_TensorRT_ConvCellExport
    ::generateConvProgramActivationDesc(Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::IActivationLayer* " << identifier
         << "_activation_layer;\n";
}

void N2D2::CPP_TensorRT_ConvCellExport
    ::generateCellProgramInstanciateLayer(Cell& cell,
                                          std::vector<std::string>& parentsName,
                                          std::ofstream& prog)
{
    generateConvProgramAddLayer(cell, parentsName, prog);

}

void N2D2::CPP_TensorRT_ConvCellExport
    ::generateConvProgramAddLayer(Cell& cell,
                                   std::vector<std::string>& parentsName,
                                   std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    std::stringstream input_name;

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        input_name << parentsName[i] << "_";

    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);
    bool isActivated = false;

    if (cellFrame != NULL) {
        if(cellFrame->getActivation()) {
            const std::string actType = cellFrame->getActivation()->getType();

            if(actType != "Linear")
                isActivated = true;
        }
    }

    std::string activationStr = isActivated ?
                                    "LayerActivation(true, " + prefix + "_ACTIVATION_TENSORRT)"
                                    : "LayerActivation(false)";

    prog << "   " << "std::vector< nvinfer1::ITensor *> "
         << identifier << "_tensor;\n";

    prog << "   " << identifier << "_tensor = " << "add_convolution(\n"
         << "       " << "\"Convolution_NATIVE_" << identifier << "\",\n"
         << "       " << activationStr << ",\n"
         << "       " << prefix << "_NB_OUTPUTS,\n"
         << "       " << prefix << "_STRIDE_X,\n"
         << "       " << prefix << "_STRIDE_Y,\n"
         << "       " << prefix << "_PADDING_X,\n"
         << "       " << prefix << "_PADDING_Y,\n"
         << "       " << prefix << "_KERNEL_WIDTH,\n"
         << "       " << prefix << "_KERNEL_HEIGHT,\n"
         << "       " << input_name.str() << "tensor,\n"
        // << "       " << identifier << "_weights_flatten,\n"
         << "       " << "mParametersPath + " << "\"weights/" << identifier << "_weights.syntxt\",\n"
         << "       " << prefix << "_WEIGHTS_SIZE,\n"
         //<< "       " << identifier << "_biases,\n"
         << "       " << "mParametersPath + " << "\"weights/" << identifier << "_bias.syntxt\",\n"
         << "       " << prefix << "_NB_OUTPUTS);\n";

}

void N2D2::CPP_TensorRT_ConvCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "   " << "add_target(" << identifier << "_tensor, "
                  << targetIdx << ");\n";
}
