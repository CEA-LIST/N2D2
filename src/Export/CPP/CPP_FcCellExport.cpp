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

void N2D2::CPP_FcCellExport::generate(FcCell& cell, const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderConstants(FcCell& cell,
                                                     std::ofstream& header)
{
    // Constants
    const unsigned int channelsSize = cell.getNbChannels()
                                      * cell.getChannelsWidth()
                                      * cell.getChannelsHeight();
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                        cell.getName()));

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

void N2D2::CPP_FcCellExport::generateHeaderFreeParameters(FcCell& cell,
                                                          std::ofstream& header)
{
    generateHeaderBias(cell, header);

    if (mThreshold > 0.0)
        generateHeaderWeightsSparse(cell, header);
    else
        generateHeaderWeights(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderBias(FcCell& cell,
                                                std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderBiasVariable(FcCell& cell,
                                                        std::ofstream& header)
{
    header << "const std::vector<WDATA_T> "
        << Utils::CIdentifier(cell.getName()) << "_biases = ";
}

void N2D2::CPP_FcCellExport::generateHeaderBiasValues(FcCell& cell,
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

void N2D2::CPP_FcCellExport::generateHeaderWeightsSparse(FcCell& cell,
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
           << "static WDATA_T " << identifier
           << "_weights_sparse[" << prefix << "_NB_WEIGHTS] = {\n";

    for (unsigned int i = 0; i < nbWeights; ++i) {
        if (i > 0)
            header << ", ";

        CellExport::generateFreeParameter(cell, weights[i], header);
    }

    header << "};\n\n";

    header << "static unsigned short " << identifier
        << "_weights_offsets[" << prefix << "_NB_WEIGHTS] = {\n";

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

void N2D2::CPP_FcCellExport::generateHeaderWeights(FcCell& cell,
                                                   std::ofstream& header)
{
    generateHeaderWeightsVariable(cell, header);
    generateHeaderWeightsValues(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderWeightsVariable(FcCell& cell,
                                                           std::ofstream
                                                           & header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_NB_WEIGHTS (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n\n";

    // Weights
    header << "const std::vector<std::vector<WDATA_T> > "
        << identifier << "_weights = \n";
}

void N2D2::CPP_FcCellExport::generateHeaderWeightsValues(FcCell& cell,
                                                       std::ofstream& header)
{
    const unsigned int channelsSize = cell.getNbChannels()
                                      * cell.getChannelsWidth()
                                      * cell.getChannelsHeight();

    header << "{\n";
    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ",\n";

        header << "    {";

        for (unsigned int channel = 0; channel < channelsSize; ++channel) {
            if (channel > 0)
                header << ", ";

            CellExport::generateFreeParameter(
                cell, cell.getWeight(output, channel), header);
        }

        header << "}";
    }

    header << "};\n\n";
}
