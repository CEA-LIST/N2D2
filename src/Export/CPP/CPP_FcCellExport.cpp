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

#include "Export/CPP/CPP_FcCellExport.hpp"

N2D2::Registrar<N2D2::FcCellExport>
N2D2::CPP_FcCellExport::mRegistrar("CPP", N2D2::CPP_FcCellExport::generate);

void N2D2::CPP_FcCellExport::generate(const FcCell& cell, const std::string& dirName) {
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/" + 
                                    Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());
    if (!header.good()) {
        throw std::runtime_error("Could not create C header file: " + fileName);
    }

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderConstants(const FcCell& cell, std::ofstream& header) {
    // Constants
    const std::size_t channelsSize = cell.getNbChannels()*cell.getChannelsWidth()* cell.getChannelsHeight();
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << channelsSize << "\n\n";


    CPP_CellExport::generateActivation(cell, header);
    CPP_CellExport::generateActivationScaling(cell, header);

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS)\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix << "_NB_CHANNELS)\n"
           << "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix << "_OUTPUTS_SIZE, " 
                                                           << prefix << "_CHANNELS_SIZE))\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT 1\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT 1\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH 1\n"
           << "#define " << prefix << "_CHANNELS_WIDTH 1\n"
           << "#define " << prefix << "_NO_BIAS " << (int) cell.getParameter<bool>("NoBias") << "\n";
}

void N2D2::CPP_FcCellExport::generateHeaderFreeParameters(const FcCell & cell, std::ofstream& header) {
    generateHeaderBias(cell, header);

    if (mThreshold > 0.0) {
        generateHeaderWeightsSparse(cell, header);
    }
    else {
        generateHeaderWeights(cell, header);
    }
}

void N2D2::CPP_FcCellExport::generateHeaderBias(const FcCell & cell, std::ofstream& header) {
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderBiasVariable(const FcCell & cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());
    header << "static const BDATA_T " << identifier << "_biases"
                << "[" << Utils::upperCase(identifier) << "_NB_OUTPUTS] = ";
}

void N2D2::CPP_FcCellExport::generateHeaderBiasValues(const FcCell & cell, std::ofstream& header) {
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    Tensor<Float_T> bias;
    
    header << "{";
    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (cell.getParameter<bool>("NoBias")) {
            header << "0";
        }
        else {
            cell.getBias(output, bias);
            CellExport::generateFreeParameter(cell, bias(0), header, Cell::Additive);
            CellExport::generateShiftScalingHalfAddition(cellFrame, output, header);
        }

        header << ", ";
    }
    header << "};\n";
}

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

    header << "#define " << prefix << "_NB_WEIGHTS " << nbWeights << "\n"
           << "static WDATA_T " << identifier
           << "_weights_sparse[" << prefix << "_NB_WEIGHTS] = {\n";

    for (std::size_t i = 0; i < nbWeights; ++i) {
        if (i > 0)
            header << ", ";

        CellExport::generateFreeParameter(cell, weights[i], header, Cell::Multiplicative);
    }

    header << "};\n\n";

    header << "static unsigned short " << identifier
        << "_weights_offsets[" << prefix << "_NB_WEIGHTS] = {\n";

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

void N2D2::CPP_FcCellExport::generateHeaderWeights(const FcCell & cell, std::ofstream& header) {
    generateHeaderWeightsVariable(cell, header);
    generateHeaderWeightsValues(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderWeightsVariable(const FcCell & cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_WEIGHTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n\n";

    // Weights
    header << "const WDATA_T " << identifier << "_weights[" << prefix << "_WEIGHTS_SIZE] = \n";
}

void N2D2::CPP_FcCellExport::generateHeaderWeightsValues(const FcCell & cell, std::ofstream& header) {
    const std::size_t channelsSize = cell.getInputsSize();

    header << "{\n";
    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << "\n";

        for (std::size_t channel = 0; channel < channelsSize; ++channel) {
            Tensor<Float_T> weight;
            cell.getWeight(output, channel, weight);

            CellExport::generateFreeParameter(cell, weight(0), header, Cell::Multiplicative);
            header << ", ";
        }
    }

    header << "};\n\n";
}
