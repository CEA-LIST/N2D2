/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "DeepNet.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/FcCell.hpp"
#include "Export/FcCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/CPP/CPP_ConvCellExport.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/CPP/CPP_FcCellExport.hpp"
#include "Export/CPP/CPP_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#include <cstdint>

N2D2::Registrar<N2D2::FcCellExport> N2D2::CPP_FcCellExport::mRegistrar(
    {"CPP", "CPP_ASMP", "CPP_STM32", "CPP_HLS"},
    N2D2::CPP_FcCellExport::generate);

N2D2::Registrar<N2D2::CPP_CellExport> N2D2::CPP_FcCellExport::mRegistrarType(
    N2D2::FcCell::Type, N2D2::CPP_FcCellExport::getInstance);

void N2D2::CPP_FcCellExport::generate(const FcCell& cell, const std::string& dirName) {
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/" + 
                                    Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());
    if (!header.good()) {
        throw std::runtime_error("Could not create CPP header file: " + fileName);
    }

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderConstants(const FcCell& cell, std::ofstream& header) {
    // Constants
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n"
           << "#define " << prefix << "_NO_BIAS " << (int) cell.getParameter<bool>("NoBias") << "\n\n";

    CPP_CellExport::generateActivation(cell, header);
    CPP_CellExport::generateWeightPrecision(cell, header);
    CPP_CellExport::generateActivationScaling(cell, header);

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*" 
                                                        << prefix << "_OUTPUTS_WIDTH*" 
                                                        << prefix << "_OUTPUTS_HEIGHT)\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix << "_NB_CHANNELS*" 
                                                         << prefix << "_CHANNELS_WIDTH*" 
                                                         << prefix << "_CHANNELS_HEIGHT)\n\n";
}

void N2D2::CPP_FcCellExport::generateHeaderFreeParameters(const FcCell & cell, std::ofstream& header) {

    if(cell.getQuantizedNbBits() > 0){
        generateHeaderBiasQAT(cell, header);
        generateHeaderWeightsQAT(cell, header);
    }
    else{
        generateHeaderBias(cell, header);

        if (mThreshold > 0.0) {
            generateHeaderWeightsSparse(cell, header);
        }
        else {
            generateHeaderWeights(cell, header);
        }
    }
}

void N2D2::CPP_FcCellExport::generateHeaderBias(const FcCell & cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());

    header << "static const BDATA_T " << identifier << "_biases[" 
               << Utils::upperCase(identifier) << "_OUTPUTS_SIZE"
           <<"] N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_BIASSES) = {";

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    Tensor<Float_T> bias;
    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (cell.getParameter<bool>("NoBias")) {
            header << "0";
        }
        else {
            cell.getBias(output, bias);
            CellExport::generateFreeParameter(bias(0), header);
        }

        CellExport::generateSingleShiftHalfAddition(cellFrame, output, header);
        header << ", ";
    }

    header << "};\n";
}

void N2D2::CPP_FcCellExport::generateHeaderBiasQAT(const FcCell & cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());

    //write explicit type
    header << "static const int32_t " << identifier << "_biases["
               << Utils::upperCase(identifier) << "_OUTPUTS_SIZE"
           <<"] N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_BIASSES) = {";

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    Tensor<Float_T> bias;
    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (cell.getParameter<bool>("NoBias")) {
            header << "0";
        }
        else {
            cell.getBias(output, bias);
            CellExport::generateFreeParameter(bias(0), header);
        }

        CellExport::generateSingleShiftHalfAddition(cellFrame, output, header);
        header << ", ";
    }

    header << "};\n";
}

void N2D2::CPP_FcCellExport::generateHeaderWeights(const FcCell & cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_WEIGHTS_SIZE (" 
               << prefix << "_OUTPUTS_SIZE*" << prefix << "_CHANNELS_SIZE" 
           << ")\n\n";

    header << "// Flatten weights with the order[OUTPUTS_SIZE][CHANNELS_SIZE]. \n"
           << "// If the previous cell was a 2D cell, CHANNELS_SIZE is flatten in "
               << "the [CHANNELS_HEIGHT][CHANNELS_WIDTH][NB_CHANNELS] order.\n";
    
    header << "static const WDATA_T " << identifier << "_weights["
               << prefix << "_WEIGHTS_SIZE"
           << "] N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_WEIGHTS) = ";

    header << "{\n";

    Tensor<Float_T> weight;
    
    // Need it in OHWC order, the order in the weights tensor is OCHW.
    std::size_t iweight = 0;
    for (std::size_t output = 0; output < cell.getNbOutputs(); output++) {
        for (std::size_t h = 0; h < cell.getChannelsHeight(); h++) {
            for (std::size_t w = 0; w < cell.getChannelsWidth(); w++) {
                for (std::size_t ch = 0; ch < cell.getNbChannels(); ch++) {
                    const std::size_t wch = ch*cell.getChannelsHeight()*cell.getChannelsWidth() + 
                                            h*cell.getChannelsWidth() + 
                                            w;
                    
                    cell.getWeight(output,  wch, weight);

                    CellExport::generateFreeParameter( weight(0), header);
                    header << ", ";

                    iweight++;
                    if(iweight % 30 == 0) {
                        header << "\n";
                    }
                }
            }
        }
    }

    header << "};\n\n";
}

void N2D2::CPP_FcCellExport::generateHeaderWeightsQAT(const FcCell & cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_WEIGHTS_SIZE ("
               << prefix << "_OUTPUTS_SIZE*" << prefix << "_CHANNELS_SIZE"
           << ")\n\n";

    header << "// Flatten weights with the order[OUTPUTS_SIZE][CHANNELS_SIZE]. \n"
           << "// If the previous cell was a 2D cell, CHANNELS_SIZE is flatten in "
               << "the [CHANNELS_HEIGHT][CHANNELS_WIDTH][NB_CHANNELS] order.\n";

    int wPrecision = (int)pow(2,std::ceil(log2(cell.getQuantizedNbBits())));
    std::string wType = "";
    bool accumulate = false;

    if((cell.getNbChannels() == 1 && (wPrecision > 0 && wPrecision < 8)) || (wPrecision == 8)){
        accumulate = false;
        wType = "int8_t";
        std::cout << Utils::cwarning << "Cell with number of channels = " << cell.getNbChannels() << ", and weight precision = " << wPrecision << " :: weights will not be accumulated!";
    }
    else if(cell.getNbChannels() > 1 && (wPrecision > 0 && wPrecision < 8)){
        accumulate = true;
        wType = "uint8_t";
    }
    else if (wPrecision > 8 && wPrecision <= 16){
        accumulate = false;
        wType = "int16_t";
        std::cout << Utils::cwarning << "Weight precision = " << wPrecision << " :: weights will not be accumulated!";
    }
    else if (wPrecision > 16 && wPrecision <= 32){
        accumulate = false;
        wType = "int32_t";
        std::cout << Utils::cwarning << "Weight precision = " << wPrecision << " :: weights will not be accumulated!";
    }
    else{
        accumulate = false;
        wType = "int64_t";
        std::cout << Utils::cwarning << "Weight precision = " << wPrecision << " :: weights will not be accumulated!";
    }


    header << "static const "<< wType << " " << identifier << "_weights["
               << prefix << "_WEIGHTS_SIZE"
           << "] N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_WEIGHTS) = ";

    header << "{\n";

    //number of int8 necessary to store
    //one weight of all channels
    std::size_t nbInt8_nbCh = 1;
    //total bit "slots"
    std::size_t nbSlot_total = 1;
    //number of channels
    std::size_t nbSlot_taken = 1;
    //number of extra 0
    std::size_t nbSlot_free = 0;
    std::size_t nbSlot_per_Int8 = 1;

    if(accumulate){
        nbInt8_nbCh = ( (cell.getNbChannels()*wPrecision) + (cell.getNbChannels()*wPrecision) % 8 ) / 8;
        nbSlot_per_Int8 = 8/(size_t)wPrecision;
        std::cout << "nbSlot_per_Int8 = " << nbSlot_per_Int8 << std::endl;
        nbSlot_total = nbInt8_nbCh*nbSlot_per_Int8;
        std::cout << " >>> nbSlot_total = " << nbSlot_total << std::endl;
        nbSlot_taken = cell.getNbChannels();
        std::cout << " >>> nbSlot_taken = " << nbSlot_taken << std::endl;
        nbSlot_free = nbSlot_total - nbSlot_taken;
        std::cout << " >>> nbSlot_free = " << nbSlot_free << std::endl;
    }
    else{
        nbSlot_total = cell.getNbChannels();
        nbSlot_taken = cell.getNbChannels();
        nbSlot_free = 0;
    }

    uint32_t mask = 0;
    if(wPrecision==4){
        mask = 0x0F;
    }
    else if(wPrecision==2){
        mask = 0x3;
    }
    else if(wPrecision==1){
        mask = 0x1;
    }

    Tensor<Float_T> weight;
    Float_T value;
    uint32_t accumulator = 0;
    int wCounter = 0;

    // Need it in OHWC order, the order in the weights tensor is OCHW.
    std::size_t iweight = 0;
    for (std::size_t output = 0; output < cell.getNbOutputs(); output++) {
        std::size_t maxHWC = (cell.getNbChannels()-1)*cell.getChannelsHeight()*cell.getChannelsWidth()
                    + (cell.getChannelsHeight()-1)*cell.getChannelsWidth()
                    + (cell.getChannelsWidth()-1);

        for (std::size_t h = 0; h < cell.getChannelsHeight(); h++) {
            for (std::size_t w = 0; w < cell.getChannelsWidth(); w++) {
                for (std::size_t ch = 0; ch < nbSlot_taken; ch++) {
                    const std::size_t wch = ch*cell.getChannelsHeight()*cell.getChannelsWidth() +
                                            h*cell.getChannelsWidth() +
                                            w;

                    if(accumulate){
                        cell.getWeight(output, wch, weight);

                        accumulator |= (static_cast<uint32_t>(std::round(weight(0))) & mask);

                        //if the last weight in accumulator
                        if(wCounter == (nbSlot_per_Int8-1)){
                            header << "0x" << std::setfill('0') << std::setw(2) << std::hex << accumulator << std::dec << ", ";
                            iweight++;
                            accumulator = 0;
                            wCounter = 0;
                            if(iweight % 30 == 0) {
                                header << "\n";
                            }
                        }
                        else{
                            accumulator <<= wPrecision;
                            ++wCounter;
                        }
                    }
                    else{

                        cell.getWeight(output,  wch, weight);
                        value = weight(0);
                        if(wPrecision == 1 && value == -1){
                            value = 0;
                        }
                        CellExport::generateFreeParameter(value, header);

                        header << ", ";
                        iweight++;
                        if(iweight % 30 == 0) {
                            header << "\n";
                        }
                    }
                }
                //fill with extra 0 if needed
                for(std::size_t free_sl = 0; free_sl < nbSlot_free; ++free_sl){
                    accumulator |= (static_cast<uint32_t>(0) & mask);
                    //if the last weight in accumulator
                    if(wCounter == (nbSlot_per_Int8-1)){
                        header << "0x" << std::setfill('0') << std::setw(2) << std::hex << accumulator << std::dec << ", ";
                        iweight++;
                        accumulator = 0;
                        wCounter = 0;
                        if(iweight % 30 == 0) {
                            header << "\n";
                        }
                    }
                    else{
                        accumulator <<= wPrecision;
                        ++wCounter;
                    }
                }
            }
        }
    }

    header << "};\n\n";
}

// Legacy function, may be removed in the future
void N2D2::CPP_FcCellExport::generateHeaderWeightsSparse(const FcCell & cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::size_t channelsSize = cell.getNbChannels()
                                      * cell.getChannelsWidth()
                                      * cell.getChannelsHeight();

    std::vector<double> weights;
    std::vector<std::size_t> offsets;
    std::size_t offset = 0;

    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        for (std::size_t channel = 0; channel < channelsSize; ++channel) {
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

    const std::size_t nbWeights = weights.size();

    header << "#define " << prefix << "_WEIGHTS_SIZE " << nbWeights << "\n"
           << "static WDATA_T " << identifier
           << "_weights_sparse[" << prefix << "_WEIGHTS_SIZE] = {\n";

    for (std::size_t i = 0; i < nbWeights; ++i) {
        if (i > 0)
            header << ", ";

        CellExport::generateFreeParameter(weights[i], header);
    }

    header << "};\n\n";

    header << "static unsigned short " << identifier
        << "_weights_offsets[" << prefix << "_WEIGHTS_SIZE] = {\n";

    for (std::size_t i = 0; i < nbWeights; ++i) {
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

std::unique_ptr<N2D2::CPP_FcCellExport>
N2D2::CPP_FcCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_FcCellExport>(new CPP_FcCellExport);
}

void N2D2::CPP_FcCellExport::generateCallCode(
    const DeepNet& deepNet,
    const Cell& cell, 
    std::stringstream& includes,
    std::stringstream& /*buffers*/, 
    std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    includes << "#include \"" << identifier << ".hpp\"\n";

    //set output type
    generateOutputType(deepNet, cell, functionCalls);

    generateBenchmarkStart(deepNet, cell, functionCalls);

    const auto& parents = deepNet.getParentCells(cell.getName());
    const std::string inputBuffer
        = Utils::CIdentifier(parents[0] ? parents[0]->getName() + "_output"
                                        : "inputs");
    const std::string outputBuffer
        = Utils::CIdentifier(cell.getName() + "_output");

    functionCalls << "    fccellPropagate<"
                << prefix << "_NB_CHANNELS, "
                << prefix << "_CHANNELS_HEIGHT, "
                << prefix << "_CHANNELS_WIDTH, "
                << prefix << "_NB_OUTPUTS, "
                << prefix << "_OUTPUTS_HEIGHT, " 
                << prefix << "_OUTPUTS_WIDTH, "
                << prefix << "_ACTIVATION, ";

    // Memory mapping: input
    const std::string parentIdentifier
        = Utils::CIdentifier((parents[0]) ? parents[0]->getName() : "env");
    const std::string parentPrefix
        = N2D2::Utils::upperCase(parentIdentifier);

    functionCalls << parentPrefix << "_MEM_CONT_OFFSET, "
        << parentPrefix << "_MEM_CONT_SIZE, "
        << parentPrefix << "_MEM_WRAP_OFFSET, "
        << parentPrefix << "_MEM_WRAP_SIZE, "
        << parentPrefix << "_MEM_STRIDE, ";

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    if (cellFrame.getActivation()
        && (cellFrame.getActivation()->getType() == "Rectifier" || cellFrame.getActivation()->getType() == "Linear"))
    {
        const Activation& activation = *cellFrame.getActivation();
        int actPrecision = (int) activation.getQuantizedNbBits();
        //if this is the last FC in the network, its activation is not quantized
        if(actPrecision > 8){
            functionCalls << "0, "
                    << prefix << "_MEM_CONT_SIZE, "
                    << prefix << "_MEM_WRAP_OFFSET, "
                    << prefix << "_MEM_WRAP_SIZE, "
                    << prefix << "_MEM_STRIDE, "
                    << prefix << "_NB_BITS_W,"
                    << CPP_CellExport::getLabelActivationRange(cell)
                << ">("
                    << inputBuffer << " , "
                    << outputBuffer << ", "
                    << identifier << "_biases, "
                    << identifier << "_weights, "
                    << CPP_CellExport::getLabelScaling(cell)
                << ");\n\n";
        }
        //if this is FC with activation quantized
        else{
            functionCalls << prefix << "_MEM_CONT_OFFSET, "
                        << prefix << "_MEM_CONT_SIZE, "
                        << prefix << "_MEM_WRAP_OFFSET, "
                        << prefix << "_MEM_WRAP_SIZE, "
                        << prefix << "_MEM_STRIDE, "
                        << prefix << "_NB_BITS_W,"
                        << CPP_CellExport::getLabelActivationRange(cell)
                    << ">("
                        << inputBuffer << " , "
                        << outputBuffer << ", "
                        << identifier << "_biases, "
                        << identifier << "_weights, "
                        << CPP_CellExport::getLabelScaling(cell)
                    << ");\n\n";
        }
    }
    else{
    // Memory mapping: output
    functionCalls << prefix << "_MEM_CONT_OFFSET, "
                << prefix << "_MEM_CONT_SIZE, "
                << prefix << "_MEM_WRAP_OFFSET, "
                << prefix << "_MEM_WRAP_SIZE, "
                << prefix << "_MEM_STRIDE, "
                << CPP_CellExport::getLabelActivationRange(cell)
            << ">("
                << inputBuffer << " , "
                << outputBuffer << ", "
                << identifier << "_biases, "
                << identifier << "_weights, "
                << CPP_CellExport::getLabelScaling(cell)
            << ");\n\n";
    }

    generateBenchmarkEnd(deepNet, cell, functionCalls);
    generateSaveOutputs(deepNet, cell, functionCalls);
}
