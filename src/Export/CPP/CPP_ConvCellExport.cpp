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
#include "Cell/ConvCell.hpp"
#include "Export/ConvCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/CPP/CPP_ConvCellExport.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/CPP/CPP_ConvCellExport.hpp"
#include "Export/CPP/CPP_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"
#include "utils/IniParser.hpp"
#include "Export/CPP/CPP_Config.hpp"

#include <fstream>
#include <string>
#include <cstdint>
#include <cmath>

N2D2::Registrar<N2D2::ConvCellExport> N2D2::CPP_ConvCellExport::mRegistrar(
    {"CPP", "CPP_ASMP", "CPP_STM32", "CPP_HLS", "CPP_Quantization"},
    N2D2::CPP_ConvCellExport::generate);

N2D2::Registrar<N2D2::CPP_CellExport>
N2D2::CPP_ConvCellExport::mRegistrarType(
        N2D2::ConvCell::Type, N2D2::CPP_ConvCellExport::getInstance);

void N2D2::CPP_ConvCellExport::generate(const ConvCell& cell, const std::string& dirName) {
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"+ 
                                    Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());
    if (!header.good()) {
        throw std::runtime_error("Could not create CPP header file: " + fileName);
    }

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);

    // Only the CPP_Quantization export can support the tools
    // for quantization aware training for now
    if (Utils::match("*CPP_Quantization*", dirName)) {
        generateHeaderFreeParameters(cell, header);
    } else {
        generateHeaderBias(cell, header);
        generateHeaderWeights(cell, header);
    }

    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderConstants(const ConvCell& cell, std::ofstream& header) {
    // Constants
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    // Handle extended padding
    std::vector<int> padding = cell.getExtendedPadding();
    padding[0] += cell.getPaddingX();  // X_L
    padding[1] += cell.getPaddingY();  // Y_T
    padding[2] += cell.getPaddingX();  // X_R
    padding[3] += cell.getPaddingY();  // Y_B

    const std::size_t oxSize = (std::size_t) (
        (cell.getChannelsWidth() + padding[0] + padding[2] - cell.getDilationX() * (cell.getKernelWidth() - 1) - 1 + cell.getStrideX())/
        static_cast<double>(cell.getStrideX())
    );

    const std::size_t oySize = (std::size_t)(
        (cell.getChannelsHeight() + padding[1] + padding[3] - cell.getDilationY() * (cell.getKernelHeight() - 1) - 1 + cell.getStrideY())/
        static_cast<double>(cell.getStrideY())
    );

    // kernel_size + (dilatation - 1) * 2
    // dilation * (kernel_size - 1) + 1
    const std::size_t dilatedKernelY = (std::size_t)(cell.getDilationY() * (cell.getKernelHeight() - 1) + 1);
    const std::size_t dilatedKernelX = (std::size_t)(cell.getDilationX() * (cell.getKernelWidth() - 1) + 1);

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_OX_SIZE " << oxSize << "\n"
           << "#define " << prefix << "_OY_SIZE " << oySize << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n"
           << "#define " << prefix << "_KERNEL_WIDTH " << cell.getKernelWidth() << "\n"
           << "#define " << prefix << "_KERNEL_HEIGHT " << cell.getKernelHeight() << "\n"
           << "#define " << prefix << "_SUB_SAMPLE_X " << cell.getSubSampleX() << "\n"
           << "#define " << prefix << "_SUB_SAMPLE_Y " << cell.getSubSampleY() << "\n"
           << "#define " << prefix << "_STRIDE_X " << cell.getStrideX() << "\n"
           << "#define " << prefix << "_STRIDE_Y " << cell.getStrideY() << "\n"
           << "#define " << prefix << "_PADDING_X " << padding[0] << "\n"
           << "#define " << prefix << "_PADDING_Y " << padding[1] << "\n"
           << "#define " << prefix << "_DILATION_X " << cell.getDilationX() << "\n"
           << "#define " << prefix << "_DILATION_Y " << cell.getDilationY() << "\n"
           << "#define " << prefix << "_NO_BIAS " << (int) cell.getParameter<bool>("NoBias") << "\n\n";

    header << "// Kernel size if dilation is included in weights\n"
           << "#define " << prefix << "_DILATED_KERNEL_WIDTH " << dilatedKernelX << "\n"
           << "#define " << prefix << "_DILATED_KERNEL_HEIGHT " << dilatedKernelY << "\n\n";

    CPP_CellExport::generateActivation(cell, header);
    CPP_CellExport::generateActivationScaling(cell, header);

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*" 
                                                        << prefix << "_OUTPUTS_WIDTH*" 
                                                        << prefix << "_OUTPUTS_HEIGHT)\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix << "_NB_CHANNELS*" 
                                                         << prefix << "_CHANNELS_WIDTH*" 
                                                         << prefix << "_CHANNELS_HEIGHT)\n\n";
}

void N2D2::CPP_ConvCellExport::generateHeaderFreeParameters(const ConvCell& cell, std::ofstream & header) {
    generateHeaderBias(cell, header);

    if (cell.getQuantizedNbBits() > 0)
        generateHeaderWeightsQAT(cell, header);
    else
        generateHeaderWeights(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderBias(const ConvCell& cell, std::ofstream& header) {
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string wType = (cell.getQuantizedNbBits() <= 0) ? "BDATA_T" :
                              (cell.getQuantizedNbBits() <= 4) ? "int16_t" :
                              (cell.getQuantizedNbBits() <= 8) ? "int32_t" :
                                                                 "int64_t";

    header << "static const " << wType << " " << identifier << "_biases[" 
           << prefix << "_NB_OUTPUTS] N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_BIASSES) = {";

    Tensor<Float_T> bias;
    for (std::size_t output = 0; output < cell.getNbOutputs(); output++) {
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

    header << "};\n\n";
}

void incrementStream(std::ofstream& header, unsigned int counter, unsigned int counterEndLine)
{
    header << ", ";
    if (counter % counterEndLine == 0)
        header << "\n";
}

void N2D2::CPP_ConvCellExport::generateHeaderWeights(const ConvCell& cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    const bool isDWConv = isDWConvolution(cell);

    // Import user options for convolution generation
    IniParser exportParams;
    if(!DeepNetExport::mExportParameters.empty())
        exportParams.load(DeepNetExport::mExportParameters);

    // If the user requires to store dilated weights
    const bool isDilatedWeights = exportParams.getProperty(
        CPP_Config::DILATED_WEIGHTS,
        CPP_Config::DILATED_WEIGHTS_DEFAULT);


    if (isDilatedWeights) {

        if(isDWConv) {
            header << "#define " << prefix << "_NB_GROUPS "
                    << cell.groupMap() << "\n"
                "#define " << prefix << "_OUTPUT_GROUP_SIZE ("
                    << prefix << "_NB_OUTPUTS / " << prefix << "_NB_GROUPS)\n"
                "#define " << prefix << "_CHANNEL_GROUP_SIZE ("
                    << prefix << "_NB_CHANNELS / " << prefix << "_NB_GROUPS)\n";

            header << "#define " << prefix << "_WEIGHTS_SIZE (" 
                                << prefix << "_NB_OUTPUTS*" 
                                << prefix << "_DILATED_KERNEL_WIDTH*" 
                                << prefix << "_DILATED_KERNEL_HEIGHT*"
                                << prefix << "_CHANNEL_GROUP_SIZE)\n\n";
        }
        else {
            header << "#define " << prefix << "_WEIGHTS_SIZE (" 
                                << prefix << "_NB_OUTPUTS*" 
                                << prefix << "_DILATED_KERNEL_WIDTH*" 
                                << prefix << "_DILATED_KERNEL_HEIGHT*"
                                << prefix << "_NB_CHANNELS)\n\n";
        }

    } else {

        if(isDWConv) {
            header << "#define " << prefix << "_NB_GROUPS "
                    << cell.groupMap() << "\n"
                "#define " << prefix << "_OUTPUT_GROUP_SIZE ("
                    << prefix << "_NB_OUTPUTS / " << prefix << "_NB_GROUPS)\n"
                "#define " << prefix << "_CHANNEL_GROUP_SIZE ("
                    << prefix << "_NB_CHANNELS / " << prefix << "_NB_GROUPS)\n";

            header << "#define " << prefix << "_WEIGHTS_SIZE (" 
                                << prefix << "_NB_OUTPUTS*" 
                                << prefix << "_KERNEL_WIDTH*" 
                                << prefix << "_KERNEL_HEIGHT*"
                                << prefix << "_CHANNEL_GROUP_SIZE)\n\n";
        }
        else {
            header << "#define " << prefix << "_WEIGHTS_SIZE (" 
                                << prefix << "_NB_OUTPUTS*" 
                                << prefix << "_KERNEL_WIDTH*" 
                                << prefix << "_KERNEL_HEIGHT*"
                                << prefix << "_NB_CHANNELS)\n\n";
        }

    }


    if(isDWConv) {
        header << "// Flatten weights with the order " 
            << "[NB_OUTPUTS][KERNEL_HEIGHT][KERNEL_WIDTH][CHANNEL_GROUP_SIZE]\n";
    } else {
        header << "// Flatten weights with the order " 
            << "[NB_OUTPUTS][KERNEL_HEIGHT][KERNEL_WIDTH][NB_CHANNELS]\n";
    }

    header << "static const WDATA_T " << identifier << "_weights["
           << prefix << "_WEIGHTS_SIZE] N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_WEIGHTS) = {";

    const Cell_Frame_Top* cellFrame
        = dynamic_cast<const Cell_Frame_Top*>(&cell);

    if (cellFrame != NULL)
        cellFrame->synchronizeToH(false);

    Tensor<Float_T> kernel;

    if (isDilatedWeights) {

        const std::size_t dilatedKernelX = (std::size_t)(cell.getDilationX() * (cell.getKernelWidth() - 1) + 1);

        std::size_t i = 0;
        for(std::size_t o = 0; o < cell.getNbOutputs(); ++o) {
            for(std::size_t sy = 0; sy < cell.getKernelHeight(); ++sy) {
                for(std::size_t sx = 0; sx < cell.getKernelWidth(); ++sx) {
                    for(std::size_t ch = 0; ch < cell.getNbChannels(); ++ch) {
                        if(isDWConv) {
                            const size_t outputGroupSize = cell.getNbOutputs() / cell.groupMap();
                            const size_t channelGroupSize = cell.getNbChannels() / cell.groupMap();
                            const size_t outputGroup = o / outputGroupSize;
                            const size_t channelGroup = ch / channelGroupSize;

                            if (outputGroup != channelGroup)
                                continue;
                        }

                        if (!cell.isConnection(ch, o)) {
                            header << "0";
                        }
                        else {
                            cell.getWeight(o, ch, kernel);

                            CellExport::generateFreeParameter(kernel(sx, sy), header);
                        }
                        i++;
                        incrementStream(header, i, 24);
                    }

                    // Add zeros for dilation X
                    if (sx != cell.getKernelWidth() - 1) {
                        for (std::size_t dilat_x = 0; dilat_x < cell.getDilationX() - 1; ++dilat_x) {
                            for(std::size_t ch = 0; ch < cell.getNbChannels(); ++ch) {
                                if(isDWConv) {
                                    const size_t outputGroupSize = cell.getNbOutputs() / cell.groupMap();
                                    const size_t channelGroupSize = cell.getNbChannels() / cell.groupMap();
                                    const size_t outputGroup = o / outputGroupSize;
                                    const size_t channelGroup = ch / channelGroupSize;

                                    if (outputGroup != channelGroup)
                                        continue;
                                }
                                header << "0";
                                i++;
                                incrementStream(header, i, 24);
                            }
                        }
                    }

                }

                // Add zeros for dilation Y
                if (sy != cell.getKernelHeight() - 1) {
                    for (std::size_t dilat_y = 0; dilat_y < cell.getDilationY() - 1; ++dilat_y) {
                        for (std::size_t sx_dilat = 0; sx_dilat < dilatedKernelX; ++sx_dilat) {
                            for(std::size_t ch = 0; ch < cell.getNbChannels(); ++ch) {
                                if(isDWConv) {
                                    const size_t outputGroupSize = cell.getNbOutputs() / cell.groupMap();
                                    const size_t channelGroupSize = cell.getNbChannels() / cell.groupMap();
                                    const size_t outputGroup = o / outputGroupSize;
                                    const size_t channelGroup = ch / channelGroupSize;

                                    if (outputGroup != channelGroup)
                                        continue;
                                }
                                header << "0";
                                i++;
                                incrementStream(header, i, 24);
                            }
                        }
                    }
                }
            }
        }

    } else {

        std::size_t i = 0;
        for(std::size_t o = 0; o < cell.getNbOutputs(); ++o) {
            for(std::size_t sy = 0; sy < cell.getKernelHeight(); ++sy) {
                for(std::size_t sx = 0; sx < cell.getKernelWidth(); ++sx) {
                    for(std::size_t ch = 0; ch < cell.getNbChannels(); ++ch) {
                        if(isDWConv) {
                            const size_t outputGroupSize = cell.getNbOutputs() / cell.groupMap();
                            const size_t channelGroupSize = cell.getNbChannels() / cell.groupMap();
                            const size_t outputGroup = o / outputGroupSize;
                            const size_t channelGroup = ch / channelGroupSize;

                            if (outputGroup != channelGroup)
                                continue;
                        }

                        if (!cell.isConnection(ch, o)) {
                            header << "0";
                        }
                        else {
                            cell.getWeight(o, ch, kernel);

                            CellExport::generateFreeParameter(kernel(sx, sy), header);
                        }
                        i++;
                        incrementStream(header, i, 24);
                    }
                }
            }
        }
    }

    if (cellFrame != NULL)
        cellFrame->keepInSync(true);

    header << "\n};\n\n";
}

void N2D2::CPP_ConvCellExport::generateHeaderWeightsQAT(const ConvCell& cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    const bool isDWConv = isDWConvolution(cell);
    if(isDWConv) {
        header << "#define " << prefix << "_NB_GROUPS "
                << cell.groupMap() << "\n"
            "#define " << prefix << "_OUTPUT_GROUP_SIZE ("
                << prefix << "_NB_OUTPUTS / " << prefix << "_NB_GROUPS)\n"
            "#define " << prefix << "_CHANNEL_GROUP_SIZE ("
                << prefix << "_NB_CHANNELS / " << prefix << "_NB_GROUPS)\n";

        header << "#define " << prefix << "_WEIGHTS_SIZE ("
                             << prefix << "_NB_OUTPUTS*"
                             << prefix << "_KERNEL_WIDTH*"
                             << prefix << "_KERNEL_HEIGHT*"
                             << prefix << "_CHANNEL_GROUP_SIZE)\n\n";

        header << "// Flatten weights with the order "
            << "[NB_OUTPUTS][KERNEL_HEIGHT][KERNEL_WIDTH][CHANNEL_GROUP_SIZE]\n";
    }
    else {
        header << "#define " << prefix << "_WEIGHTS_SIZE ("
                             << prefix << "_NB_OUTPUTS*"
                             << prefix << "_KERNEL_WIDTH*"
                             << prefix << "_KERNEL_HEIGHT*"
                             << prefix << "_NB_CHANNELS)\n\n";

        header << "// Flatten weights with the order "
            << "[NB_OUTPUTS][KERNEL_HEIGHT][KERNEL_WIDTH][NB_CHANNELS]\n";
    }

    // Force wPrecision to be a multiple of 2 (for example, 3 bits weights will
    // be stored on 4 bits)
    const unsigned int wPrecision
        = (int)std::pow(2, std::ceil(std::log2(cell.getQuantizedNbBits())));
    const bool accumulate = (cell.getNbChannels() > 1
                                && (wPrecision > 0 && wPrecision < 8));

    header << "static const data<" << wPrecision << "> " << identifier << "_weights["
           << prefix << "_WEIGHTS_SIZE] N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_WEIGHTS) = {";


    //number of int8 necessary to store
    //one weight of all channels, when !isDWConv
    std::size_t nbInt8_nbCh = 1;
    //total bit "slots"
    std::size_t nbSlot_total_nbCh = 1;
    //number of channels
    std::size_t nbSlot_taken_nbCh = 1;
    //number of extra 0
    std::size_t nbSlot_free_nbCh = 0;

    //when isDWConv
    std::size_t nbInt8_kWidth = 1;
    //total bit "slots"
    std::size_t nbSlot_total_kWidth = 1;
    //number of channels
    std::size_t nbSlot_taken_kWidth = 1;
    //number of extra 0
    std::size_t nbSlot_free_kWidth = 0;

    std::size_t nbSlot_per_Int8 = 1;

    if(accumulate && !isDWConv){
        nbInt8_nbCh = ( (cell.getNbChannels()*wPrecision) + (cell.getNbChannels()*wPrecision) % 8 ) / 8;
        nbSlot_per_Int8 = 8/(size_t)wPrecision;
        nbSlot_total_nbCh = nbInt8_nbCh*nbSlot_per_Int8;
        nbSlot_taken_nbCh = cell.getNbChannels();
        nbSlot_free_nbCh = nbSlot_total_nbCh - nbSlot_taken_nbCh;

        nbSlot_total_kWidth = cell.getKernelWidth();
        nbSlot_taken_kWidth = cell.getKernelWidth();
        nbSlot_free_kWidth = 0;
    }
    else if(accumulate && isDWConv){
        nbInt8_kWidth = ( (cell.getKernelWidth()*wPrecision) + (cell.getKernelWidth()*wPrecision) % 8 ) / 8;
        nbSlot_per_Int8 = 8/(size_t)wPrecision;
        nbSlot_total_kWidth = nbInt8_kWidth*nbSlot_per_Int8;
        nbSlot_taken_kWidth = cell.getKernelWidth();
        nbSlot_free_kWidth = nbSlot_total_kWidth - nbSlot_taken_kWidth;

        nbSlot_total_nbCh = cell.getNbChannels();
        nbSlot_taken_nbCh = cell.getNbChannels();
        nbSlot_free_nbCh = 0;
    }
    else{
        nbSlot_total_nbCh = cell.getNbChannels();
        nbSlot_taken_nbCh = cell.getNbChannels();
        nbSlot_free_nbCh = 0;

        nbSlot_total_kWidth = cell.getKernelWidth();
        nbSlot_taken_kWidth = cell.getKernelWidth();
        nbSlot_free_kWidth = 0;
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

    Tensor<Float_T> kernel;
    Float_T value;
    uint32_t accumulator = 0;
    // uint32_t accumulatorDW = 0;  // Not used for now
    std::size_t wCounter = 0;
    std::size_t i = 0;

    for(std::size_t o = 0; o < cell.getNbOutputs(); ++o) {
        for(std::size_t sy = 0; sy < cell.getKernelHeight(); ++sy) {
            for(std::size_t sx = 0; sx < nbSlot_taken_kWidth; ++sx) {
                for(std::size_t real_sl = 0; real_sl < nbSlot_taken_nbCh; ++real_sl){
                    if(isDWConv) {
                        const size_t outputGroupSize = cell.getNbOutputs() / cell.groupMap();
                        const size_t channelGroupSize = cell.getNbChannels() / cell.groupMap();
                        const size_t outputGroup = o / outputGroupSize;
                        const size_t channelGroup = real_sl / channelGroupSize;
                        if (outputGroup != channelGroup)
                            continue;
                    }

                    if(accumulate){
                        if (!cell.isConnection(real_sl, o)) {
                            accumulator |= (static_cast<uint32_t>(0) & mask);
                        }
                        else {
                            cell.getWeight(o, real_sl, kernel);
                            value = kernel(sx, sy);
                            if(wPrecision == 1 && value == -1){
                                value = 0;
                            }
                            accumulator |= (static_cast<uint32_t>(std::round(value)) & mask);
                        }

                        //if the last weight in accumulator
                        if(wCounter == (nbSlot_per_Int8-1)){
                            //if uint8_t for accumulator, the result : 0x0ó, 0x0Ï, 0x03, 0x0ð, 0x0ü, ...
                            //https://stackoverflow.com/questions/23575381/behavior-of-cout-hex-with-uint8-and-uint16
                            //static_cast<int> or + before accumulator
                            //or use uint32_t for accumulator type
                            header << "0x" << std::setfill('0') << std::setw(2) << std::hex << accumulator << std::dec << ", ";
                            i++;
                            accumulator = 0;
                            wCounter = 0;
                            if(i % 24 == 0) {
                                header << "\n";
                            }
                        }
                        else{
                            accumulator <<= wPrecision;
                            ++wCounter;
                        }
                    }
                    else{
                        if (!cell.isConnection(real_sl, o)) {
                            header << "0";
                        }
                        else {
                            cell.getWeight(o, real_sl, kernel);
                            CellExport::generateFreeParameter(kernel(sx, sy), header);
                        }

                        header << ", ";
                        i++;

                        if(i % 24 == 0) {
                            header << "\n";
                        }
                    }

                }

                //conv :: fill with extra 0 if needed
                for(std::size_t free_sl = 0; free_sl < nbSlot_free_nbCh; ++free_sl){

                    accumulator |= (static_cast<uint32_t>(0) & mask);

                    //if the last weight in accumulator
                    if(wCounter == (nbSlot_per_Int8-1)){
                        header << "0x" << std::setfill('0') << std::setw(2) << std::hex << accumulator << std::dec << ", ";
                        i++;
                        accumulator = 0;
                        wCounter = 0;
                        if(i % 24 == 0) {
                            header << "\n";
                        }
                    }
                    else{
                        accumulator <<= wPrecision;
                        ++wCounter;
                    }
                }
            }

            //convDW :: fill with extra 0 if needed for
            for(std::size_t free_sl = 0; free_sl < nbSlot_free_kWidth; ++free_sl) {

                accumulator |= (static_cast<uint32_t>(0) & mask);

                //if the last weight in accumulator
                if(wCounter == (nbSlot_per_Int8-1)){
                    header << "0x" << std::setfill('0') << std::setw(2) << std::hex << accumulator << std::dec << ", ";
                    i++;
                    accumulator = 0;
                    wCounter = 0;
                    if(i % 24 == 0) {
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

    header << "\n};\n\n";
}

bool N2D2::CPP_ConvCellExport::isDWConvolution(const Cell& cell) {
    return cell.groupMap() > 1; //TODO 
}

std::unique_ptr<N2D2::CPP_ConvCellExport>
N2D2::CPP_ConvCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_ConvCellExport>(new CPP_ConvCellExport);
}

void N2D2::CPP_ConvCellExport::generateCallCode(
    const DeepNet& deepNet,
    const Cell& cell, 
    std::stringstream& includes,
    std::stringstream& /*buffers*/, 
    std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    includes << "#include \"" << identifier << ".hpp\"\n";

    generateBenchmarkStart(deepNet, cell, functionCalls);

    const auto& parents = deepNet.getParentCells(cell.getName());
    const std::string inputBuffer
        = Utils::CIdentifier(parents[0] ? parents[0]->getName() + "_output"
                                        : "inputs");
    const std::string outputBuffer
        = Utils::CIdentifier(cell.getName() + "_output");

    // Import user options for convolution generation
    IniParser exportParams;
    if(!DeepNetExport::mExportParameters.empty())
        exportParams.load(DeepNetExport::mExportParameters);

    // If the user requires to store dilated weights
    const bool isDilatedWeights = exportParams.getProperty(
        CPP_Config::DILATED_WEIGHTS,
        CPP_Config::DILATED_WEIGHTS_DEFAULT);

    if (cell.getType() == ConvCell::Type) {
        const auto convCell = std::dynamic_pointer_cast<ConvCell>(deepNet.getCell(cell.getName()));

        // If the user requires to store dilated weights
        // No need to use dilated convolution kernels
        if (isDilatedWeights) {

            if(CPP_ConvCellExport::isDWConvolution(cell))
                functionCalls << "    convcellDWPropagate";
            else
                functionCalls << "    convcellPropagate";

            functionCalls << "<"
                        << prefix << "_NB_CHANNELS, "
                        << prefix << "_CHANNELS_HEIGHT, "
                        << prefix << "_CHANNELS_WIDTH, "
                        << prefix << "_NB_OUTPUTS, "
                        << prefix << "_OUTPUTS_HEIGHT, " 
                        << prefix << "_OUTPUTS_WIDTH, "
                        << prefix << "_PADDING_Y, "
                        << prefix << "_PADDING_X, "
                        << prefix << "_STRIDE_Y, "
                        << prefix << "_STRIDE_X, "
                        << prefix << "_DILATED_KERNEL_HEIGHT, "
                        << prefix << "_DILATED_KERNEL_WIDTH, "
                        << prefix << "_ACTIVATION, ";

        }
        else {

            if (convCell->getDilationY() != 1 || convCell->getDilationX() != 1) {

                if(CPP_ConvCellExport::isDWConvolution(cell))
                    functionCalls << "    convcellDWPropagate";
                else
                    functionCalls << "    convcellPropagate";

                functionCalls << "<"
                            << prefix << "_NB_CHANNELS, "
                            << prefix << "_CHANNELS_HEIGHT, "
                            << prefix << "_CHANNELS_WIDTH, "
                            << prefix << "_NB_OUTPUTS, "
                            << prefix << "_OUTPUTS_HEIGHT, " 
                            << prefix << "_OUTPUTS_WIDTH, "
                            << prefix << "_PADDING_Y, "
                            << prefix << "_PADDING_X, "
                            << prefix << "_STRIDE_Y, "
                            << prefix << "_STRIDE_X, "
                            << prefix << "_DILATION_Y, "
                            << prefix << "_DILATION_X, "
                            << prefix << "_KERNEL_HEIGHT, "
                            << prefix << "_KERNEL_WIDTH, "
                            << prefix << "_ACTIVATION, ";
            } 
            else {

                if(CPP_ConvCellExport::isDWConvolution(cell))
                    functionCalls << "    convcellDWPropagate";
                else
                    functionCalls << "    convcellPropagate";

                functionCalls << "<"
                            << prefix << "_NB_CHANNELS, "
                            << prefix << "_CHANNELS_HEIGHT, "
                            << prefix << "_CHANNELS_WIDTH, "
                            << prefix << "_NB_OUTPUTS, "
                            << prefix << "_OUTPUTS_HEIGHT, " 
                            << prefix << "_OUTPUTS_WIDTH, "
                            << prefix << "_PADDING_Y, "
                            << prefix << "_PADDING_X, "
                            << prefix << "_STRIDE_Y, "
                            << prefix << "_STRIDE_X, "
                            << prefix << "_KERNEL_HEIGHT, "
                            << prefix << "_KERNEL_WIDTH, "
                            << prefix << "_ACTIVATION, ";
            }
        }
    }

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

    // Memory mapping: output
    functionCalls << prefix << "_MEM_CONT_OFFSET, "
                << prefix << "_MEM_CONT_SIZE, "
                << prefix << "_MEM_WRAP_OFFSET, "
                << prefix << "_MEM_WRAP_SIZE, "
                << prefix << "_MEM_STRIDE"
            << ">"
            <<"("
                << inputBuffer << " , "
                << outputBuffer << ", "
                << identifier << "_biases, "
                << identifier << "_weights, "
                << prefix << "_SCALING"
            << ");\n\n";

    generateBenchmarkEnd(deepNet, cell, functionCalls);
    generateSaveOutputs(deepNet, cell, functionCalls);
}
