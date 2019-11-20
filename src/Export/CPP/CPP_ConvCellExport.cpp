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

#include "Cell/Cell.hpp"
#include "Cell/ConvCell.hpp"
#include "Export/CPP/CPP_ConvCellExport.hpp"
#include "utils/Registrar.hpp"

#include <fstream>
#include <string>

N2D2::Registrar<N2D2::ConvCellExport>
N2D2::CPP_ConvCellExport::mRegistrar("CPP", N2D2::CPP_ConvCellExport::generate);

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
    const std::size_t oxSize = (std::size_t) (
        (cell.getChannelsWidth() + 2*cell.getPaddingX() - cell.getKernelWidth() + cell.getStrideX())/
        static_cast<double>(cell.getStrideX())
    );

    const std::size_t oySize = (std::size_t)(
        (cell.getChannelsHeight() + 2*cell.getPaddingY() - cell.getKernelHeight() + cell.getStrideY())/
        static_cast<double>(cell.getStrideY())
    );
    
        
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

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
           << "#define " << prefix << "_PADDING_X " << cell.getPaddingX() << "\n"
           << "#define " << prefix << "_PADDING_Y " << cell.getPaddingY() << "\n"
           << "#define " << prefix << "_NO_BIAS " << (int) cell.getParameter<bool>("NoBias") << "\n\n";


    CPP_CellExport::generateActivation(cell, header);
    CPP_CellExport::generateActivationScaling(cell, header);

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*" 
                                                        << prefix << "_OUTPUTS_WIDTH*" 
                                                        << prefix << "_OUTPUTS_HEIGHT)\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix << "_NB_CHANNELS*" 
                                                         << prefix << "_CHANNELS_WIDTH*" 
                                                         << prefix << "_CHANNELS_HEIGHT)\n"
           << "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix << "_OUTPUTS_SIZE, " 
                                                           << prefix << "_CHANNELS_SIZE))\n\n";
}

void N2D2::CPP_ConvCellExport::generateHeaderFreeParameters(const ConvCell& cell, std::ofstream & header) {
    generateHeaderBias(cell, header);
    generateHeaderWeights(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderBias(const ConvCell& cell, std::ofstream& header) {
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
    header << "\n";
}

void N2D2::CPP_ConvCellExport::generateHeaderBiasVariable(const ConvCell& cell, std::ofstream& header) {
    const std::string indentifier = Utils::CIdentifier(cell.getName());
    header << "static const BDATA_T " << indentifier << "_biases"
                << "[" << Utils::upperCase(indentifier) << "_NB_OUTPUTS] = ";
}

void N2D2::CPP_ConvCellExport::generateHeaderBiasValues(const ConvCell& cell, std::ofstream& header) {
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    Tensor<Float_T> bias;
    
    header << "{";
    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (cell.getParameter<bool>("NoBias")) {
            header << "0";
        }
        else {
            cell.getBias(output, bias);
            CellExport::generateFreeParameter(bias(0), header, true);
        }

        CellExport::generateSingleShiftHalfAddition(cellFrame, output, header);
        header << ", ";
    }
    header << "};\n";
}

void N2D2::CPP_ConvCellExport::generateHeaderWeights(const ConvCell& cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);


    header << "#define " << prefix << "_WEIGHTS_SIZE (" 
           << prefix << "_NB_OUTPUTS*" 
           << prefix << "_NB_CHANNELS*" 
           << prefix << "_KERNEL_WIDTH*" 
           << prefix << "_KERNEL_HEIGHT)\n\n";

    header << "// Flatten weights with the order[NB_OUTPUTS][NB_CHANNELS][KERNEL_WIDTH][KERNEL_HEIGHT]\n";
    header << "static const WDATA_T " << identifier << "_weights["
           << prefix << "_WEIGHTS_SIZE] = {";

    std::size_t i = 0;
    for(std::size_t o = 0; o < cell.getNbOutputs(); ++o) {
        for(std::size_t ch = 0; ch < cell.getNbChannels(); ++ch) {
            Tensor<Float_T> kernel;
            cell.getWeight(o, ch, kernel);

            for(std::size_t sy = 0; sy < cell.getKernelHeight(); ++sy) {
                for(std::size_t sx = 0; sx < cell.getKernelWidth(); ++sx) {
                    if(o > 0 || ch > 0 || sy > 0 || sx > 0) {
                        header << ", ";
                    }

                    if(i % 24 == 0) {
                        header << "\n";
                    }

                    if(!cell.isConnection(ch, o)) {
                        header << "0";
                    }
                    else {
                        CellExport::generateFreeParameter(kernel(sx, sy), header);
                    }

                    i++;
                }
            }
        }
    }

    header << "\n};\n\n";
}
