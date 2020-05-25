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

#include "Export/CPP_TensorRT/CPP_TensorRT_FcCellExport.hpp"

N2D2::Registrar<N2D2::FcCellExport> N2D2::CPP_TensorRT_FcCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_FcCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_FcCellExport::mRegistrarType(
    FcCell::Type, N2D2::CPP_TensorRT_FcCellExport::getInstance);

void N2D2::CPP_TensorRT_FcCellExport::generate(FcCell& cell,
                                            const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";
    const std::string weightName = dirName + "/dnn/weights/"
        + Utils::CIdentifier(cell.getName()) + "_weights.syntxt";
    const std::string biasName = dirName + "/dnn/weights/"
        + Utils::CIdentifier(cell.getName()) + "_bias.syntxt";

    std::ofstream weights(weightName.c_str());
    std::ofstream bias(biasName.c_str());
    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_TensorRT_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);

    generateFileBias(cell, bias);
    bias.close();

    generateFileWeights(cell, weights);
    weights.close();

}

void N2D2::CPP_TensorRT_FcCellExport::generateHeaderConstants(FcCell& cell,
                                                           std::ofstream
                                                           & header)
{
    // Constants
    const unsigned int channelsSize = cell.getInputsSize();
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


void N2D2::CPP_TensorRT_FcCellExport::generateHeaderFreeParameters(FcCell& cell,
                                                                std::ofstream
                                                                & header)
{
    //generateHeaderBias(cell, header);

    if (mThreshold > 0.0)
        generateHeaderWeightsSparse(cell, header);
    else
        generateHeaderWeights(cell, header);
}

void N2D2::CPP_TensorRT_FcCellExport::generateHeaderBias(FcCell& cell,
                                                      std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_TensorRT_FcCellExport::generateHeaderBiasVariable(FcCell& cell,
                                                              std::ofstream
                                                              & header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    header << "static BDATA_T " << identifier << "_biases["
           << Utils::upperCase(identifier) << "_NB_OUTPUTS] = ";
}

void N2D2::CPP_TensorRT_FcCellExport::generateHeaderBiasValues(FcCell& cell,
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

            CellExport::generateFreeParameter(bias(0), header);
        }
    }

    header << "};\n";
}

void N2D2::CPP_TensorRT_FcCellExport::generateHeaderWeights(FcCell& cell,
                                                         std::ofstream& header)
{
    //const unsigned int channelsSize = cell.getInputsSize();
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_NB_WEIGHTS (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n\n";
/*
    // Weights flatten
    header << "#define " << prefix << "_WEIGHTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n"
           << "static WDATA_T " << identifier << "_weights_flatten["
           << prefix << "_WEIGHTS_SIZE] = {\n";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < channelsSize; ++channel) {
            if (output > 0 || channel > 0)
                header << ", ";

            CellExport::generateFreeParameter(cell.getWeight(output, channel), header);
        }
    }

    header << "};\n\n";
    */
}

void N2D2::CPP_TensorRT_FcCellExport::generateFileBias(FcCell& cell,
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

void N2D2::CPP_TensorRT_FcCellExport::generateFileWeights(FcCell& cell,
                                                           std::ofstream& file)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const unsigned int channelsSize = cell.getNbChannels()
                                      * cell.getChannelsWidth()
                                      * cell.getChannelsHeight();

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < channelsSize; ++channel) {
            if (output > 0 || channel > 0)
                file << " ";

            Tensor<Float_T> weight;
            cell.getWeight(output, channel, weight);

            CellExport::generateFreeParameter(weight(0), file, false);
        }
    }


}


void N2D2::CPP_TensorRT_FcCellExport::generateHeaderWeightsSparse(FcCell& cell,
                                                       std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const unsigned int channelsSize = cell.getInputsSize();

    std::vector<double> weights;
    std::vector<unsigned int> offsets;
    unsigned int offset = 0;

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < channelsSize; ++channel) {
            Tensor<double> weight;
            cell.getWeight(output, channel, weight);

            if (std::fabs(weight(0)) >= mThreshold) {
                weights.push_back(weight(0));
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

        CellExport::generateFreeParameter(weights[i], header);
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

std::unique_ptr<N2D2::CPP_TensorRT_FcCellExport>
N2D2::CPP_TensorRT_FcCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_TensorRT_FcCellExport>(new CPP_TensorRT_FcCellExport);
}

void N2D2::CPP_TensorRT_FcCellExport
    ::generateCellProgramDescriptors(Cell& cell, std::ofstream& prog)
{

    generateFcProgramTensorDesc(cell, prog);
    generateFcProgramLayerDesc(cell, prog);
    generateFcProgramFilterDesc(cell, prog);
    generateFcProgramActivationDesc(cell, prog);
}

void N2D2::CPP_TensorRT_FcCellExport
    ::generateFcProgramTensorDesc(Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::ITensor* " << identifier << "_tensor;\n";
}

void N2D2::CPP_TensorRT_FcCellExport
    ::generateFcProgramLayerDesc(Cell& cell,std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::IFullyConnectedLayer* " << identifier << "_layer;\n";
}

void N2D2::CPP_TensorRT_FcCellExport
    ::generateFcProgramFilterDesc(Cell& cell,std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::Weights " << identifier
         << "_filter;\n";
    prog << "nvinfer1::Weights " << identifier
         << "_bias;\n";
}

void N2D2::CPP_TensorRT_FcCellExport
    ::generateFcProgramActivationDesc(Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::IActivationLayer* " << identifier
         << "_activation_layer;\n";
}

void N2D2::CPP_TensorRT_FcCellExport
    ::generateCellProgramInstanciateLayer( Cell& cell,
                                           std::vector<std::string>& parentsName,
                                           std::ofstream& prog)
{
    generateFcProgramAddLayer(cell, parentsName, prog);
}

void N2D2::CPP_TensorRT_FcCellExport
    ::generateFcProgramAddLayer(Cell& cell,
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

    prog << "   " << identifier << "_tensor = " << "add_fc(tsrRTHandles.netDef.back(),\n"
         << "       " << "tsrRTHandles.netBuilder,\n"
         << "       " << "tsrRTHandles.dT,\n"
         << "       " << "\"FullyConnected_NATIVE_" << identifier << "\",\n"
         << "       " << activationStr << ",\n"
         << "       " << prefix << "_NB_OUTPUTS,\n"
         << "       " << input_name.str() << "tensor,\n"
         //<< "       " << identifier << "_weights_flatten,\n"
         << "       " << "\"dnn/weights/" << identifier << "_weights.syntxt\",\n"
         << "       " << prefix << "_NB_WEIGHTS,\n"
         << "       " << "\"dnn/weights/" << identifier << "_bias.syntxt\");\n";

}

void N2D2::CPP_TensorRT_FcCellExport
    ::generateCellProgramAllocateMemory(unsigned int targetIdx, std::ofstream& prog)
{
    prog << "   " << "CHECK_CUDA_STATUS( cudaMalloc(&inout_buffer["
                  << targetIdx + 1 << "], " // Added 1 for stride the input buffer
                  << "sizeof(DATA_T)*batchSize"
                  << "*NB_OUTPUTS[" << targetIdx << "]"
                  << "));\n";
}

void N2D2::CPP_TensorRT_FcCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "   " << "add_target(tsrRTHandles.netDef.back(), " << identifier << "_tensor, "
                  << targetIdx << ");\n";

}
