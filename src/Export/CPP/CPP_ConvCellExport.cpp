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

#include <fstream>
#include <string>
#include <cstdint>

N2D2::Registrar<N2D2::ConvCellExport> N2D2::CPP_ConvCellExport::mRegistrar(
    {"CPP", "CPP_ASMP", "CPP_STM32", "CPP_HLS"},
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
    generateHeaderFreeParameters(cell, header);
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
        (cell.getChannelsWidth() + padding[0] + padding[2] - cell.getKernelWidth() + cell.getStrideX())/
        static_cast<double>(cell.getStrideX())
    );

    const std::size_t oySize = (std::size_t)(
        (cell.getChannelsHeight() + padding[1] + padding[3] - cell.getKernelHeight() + cell.getStrideY())/
        static_cast<double>(cell.getStrideY())
    );

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

void N2D2::CPP_ConvCellExport::generateHeaderFreeParameters(const ConvCell& cell, std::ofstream & header) {
    if(cell.getQuantizedNbBits() > 0){
        generateHeaderBiasQAT(cell, header);
        generateHeaderWeightsQAT(cell, header);
    }
    else{
        generateHeaderBias(cell, header);
        generateHeaderWeights(cell, header);
    }
}

void N2D2::CPP_ConvCellExport::generateHeaderBias(const ConvCell& cell, std::ofstream& header) {
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    
    header << "static const BDATA_T " << identifier << "_biases[" 
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

void N2D2::CPP_ConvCellExport::generateHeaderBiasQAT(const ConvCell& cell, std::ofstream& header) {
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    //write explicit type, int32 for now
    header << "static const int32_t " << identifier << "_biases["
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

void N2D2::CPP_ConvCellExport::generateHeaderWeights(const ConvCell& cell, std::ofstream& header) {
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

    header << "static const WDATA_T " << identifier << "_weights["
           << prefix << "_WEIGHTS_SIZE] N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_WEIGHTS) = {";

    Tensor<Float_T> kernel;

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

                    header << ", ";

                    i++;
                    if(i % 24 == 0) {
                        header << "\n";
                    }
                }
            }
        }
    }

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

    //write explicit type : int8_t is default type for weights in 8,4,2,1 bit precision
    int wPrecision = (int)cell.getQuantizedNbBits();
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
    else{
        accumulate = false;
        wType = "int32_t";
        std::cout << Utils::cwarning << "Weight precision (" << wPrecision << " in cell " << cell.getName() << ", is not supported !" << Utils::cdef << std::endl;
    }

    header << "static const " << wType << " " << identifier << "_weights["
           << prefix << "_WEIGHTS_SIZE] N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_WEIGHTS) = {";


    Tensor<Float_T> kernel;
    uint8_t accumulator = 0;
    std::size_t i = 0;

    //channel-precision alignement
    std::size_t nbCh_prec_align = 1;
    //number of channels in accumulator
    std::size_t nbCh_per_acc = 1;

    if(accumulate){
        nbCh_prec_align = ( (cell.getNbChannels()*wPrecision) + (cell.getNbChannels()*wPrecision) % 8 ) / 8;
        nbCh_per_acc = cell.getNbChannels() / nbCh_prec_align + (cell.getNbChannels() % nbCh_prec_align);
        //std::cout << "nb channel = " << cell.getNbChannels() << std::endl;
        //std::cout << "nbCh_prec_align = " << nbCh_prec_align << std::endl;
        //std::cout << "nbCh_per_acc = " << nbCh_per_acc << std::endl;
        //number of weights per accumulator : 2, 4, 8 for 4b, 2b and 1b respectively
        assert(nbCh_per_acc == 8/(size_t)wPrecision);
    }
    else{
        nbCh_prec_align = cell.getNbChannels();
    }

    //int extraWeight = 0;

    for(std::size_t o = 0; o < cell.getNbOutputs(); ++o) {
        for(std::size_t sy = 0; sy < cell.getKernelHeight(); ++sy) {
            for(std::size_t sx = 0; sx < cell.getKernelWidth(); ++sx) {
                for(std::size_t ch_pr = 0; ch_pr < nbCh_prec_align; ++ch_pr) {

                    //std::cout << "nbCh_prec_align = " << nbCh_prec_align << std::endl;
                    //std::cout << "ch_pr = " << ch_pr << std::endl;

                    if(isDWConv) {
                        const size_t outputGroupSize = cell.getNbOutputs() / cell.groupMap();
                        const size_t channelGroupSize = cell.getNbChannels() / cell.groupMap();
                        const size_t outputGroup = o / outputGroupSize;
                        const size_t channelGroup = ch_pr / channelGroupSize;
                        //const size_t channelGroup = ch / channelGroupSize;

                        if (outputGroup != channelGroup)
                            continue;
                    }

                    //accumulate weights
                    if(accumulate){
                        if (!cell.isConnection(ch_pr, o)) {
                            //TODO: adapt dw to accumulator
                            header << "0";
                        }
                        else {
                            //channels per accumulator
                            for(std::size_t ch_in_acc = 0; ch_in_acc < nbCh_per_acc; ++ch_in_acc) {
                                std::cout << "nbCh_per_acc = " << nbCh_per_acc << std::endl;
                                std::cout << "ch_in_acc = " << ch_in_acc << std::endl;
                                //channel number
                                std::size_t ch = ch_pr*nbCh_per_acc+ch_in_acc;

                                //std::cout << " o = " << o << " sy = " << sy << " sx = " << sx << "ch_pr = "<< ch_pr << " ch = " << ch << std::endl;

                                //if this is an "extra" channel, fill it with 0
                                //good place to do it? we insert 0th in the middle of "line"
                                //insert all 0th at once at the end of the "line": sx = cell.getKernelWidth()-1 ?
                                if(ch > cell.getNbChannels()-1){
                                    //std::cout << "extra channel" << std::endl;
                                    kernel.resize({cell.getKernelWidth(),cell.getKernelHeight()});
                                    kernel.fill(0.0);
                                }
                                else{
                                    cell.getWeight(o, ch, kernel);
                                }

                                accumulator |= (static_cast<uint8_t>(std::round(kernel(sx, sy))) & 0x0F);

                                //shift if not the last weight in accumulator
                                if(ch_in_acc < (nbCh_per_acc-1)){
                                    accumulator <<= wPrecision;
                                }
                            }

                            //write to header
                            //std::cout << " accumulated value = " << +accumulator << std::endl;
                            header << +accumulator << ", ";
                            i++;
                            accumulator = 0;
                            if(i % 24 == 0) {
                                header << "\n";
                            }
                        }

                    }
                    //do not accumulate weights
                    else{
                        if (!cell.isConnection(ch_pr, o)) {
                            header << "0";
                        }
                        else {
                            std::cout << " o = " << o << " sx = " << sx << " sy = " << sy << " ch = " << ch_pr << std::endl;
                            cell.getWeight(o, ch_pr, kernel);
                            CellExport::generateFreeParameter(kernel(sx, sy), header);
                        }
                        header << ", ";
                        i++;

                        if(i % 24 == 0) {
                            header << "\n";
                        }
                    }
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

    //set output type
    generateOutputType(deepNet, cell, functionCalls);

    generateBenchmarkStart(deepNet, cell, functionCalls);

    const auto& parents = deepNet.getParentCells(cell.getName());
    const std::string inputBuffer
        = Utils::CIdentifier(parents[0] ? parents[0]->getName() + "_output"
                                        : "inputs");
    const std::string outputBuffer
        = Utils::CIdentifier(cell.getName() + "_output");

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
                << prefix << "_MEM_STRIDE,"
                << prefix << "_NB_BITS_W,"
                << CPP_CellExport::getLabelActivationRange(cell)
            << ">"
            <<"("
                << inputBuffer << " , "
                << outputBuffer << ", "
                << identifier << "_biases, "
                << identifier << "_weights, "
                << CPP_CellExport::getLabelScaling(cell)
            << ");\n\n";

    generateBenchmarkEnd(deepNet, cell, functionCalls);
    generateSaveOutputs(deepNet, cell, functionCalls);
}
