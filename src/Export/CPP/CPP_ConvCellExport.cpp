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

#include "Export/CPP/CPP_ConvCellExport.hpp"

N2D2::Registrar<N2D2::ConvCellExport>
N2D2::CPP_ConvCellExport::mRegistrar("CPP", N2D2::CPP_ConvCellExport::generate);

void N2D2::CPP_ConvCellExport::generate(ConvCell& cell,
                                        const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include"
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

void N2D2::CPP_ConvCellExport::generateHeaderConstants(ConvCell& cell,
                                                            std::ofstream
                                                            & header)
{
    // Constants
    const unsigned int oxSize
        = (unsigned int)((cell.getChannelsWidth() + 2 * cell.getPaddingX()
                          - cell.getKernelWidth() + cell.getStrideX())
                         / (double)cell.getStrideX());
    const unsigned int oySize
        = (unsigned int)((cell.getChannelsHeight() + 2 * cell.getPaddingY()
                          - cell.getKernelHeight() + cell.getStrideY())
                         / (double)cell.getStrideY());
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                        cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs()
           << "\n"
              "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels()
           << "\n"
              "#define " << prefix << "_OUTPUTS_WIDTH "
           << cell.getOutputsWidth() << "\n"
                                        "#define " << prefix
           << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
                                                               "#define "
           << prefix << "_OX_SIZE " << oxSize << "\n"
                                                 "#define " << prefix
           << "_OY_SIZE " << oySize << "\n"
                                       "#define " << prefix
           << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
                                                               "#define "
           << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight()
           << "\n"
              "#define " << prefix << "_KERNEL_WIDTH " << cell.getKernelWidth()
           << "\n"
              "#define " << prefix << "_KERNEL_HEIGHT "
           << cell.getKernelHeight() << "\n"
                                        "#define " << prefix << "_SUB_SAMPLE_X "
           << cell.getSubSampleX() << "\n"
                                      "#define " << prefix << "_SUB_SAMPLE_Y "
           << cell.getSubSampleY() << "\n"
                                      "#define " << prefix << "_STRIDE_X "
           << cell.getStrideX() << "\n"
                                   "#define " << prefix << "_STRIDE_Y "
           << cell.getStrideY() << "\n"
                                   "#define " << prefix << "_PADDING_X "
           << cell.getPaddingX() << "\n"
                                    "#define " << prefix << "_PADDING_Y "
           << cell.getPaddingY() << "\n"
                                    "#define " << prefix << "_NO_BIAS "
           << (cell.getParameter<bool>("NoBias") ? "1" : "0") << "\n\n";

    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);

    if (cellFrame != NULL) {
        header << "#define " << prefix << "_ACTIVATION "
               << ((cellFrame->getActivation())
                       ? cellFrame->getActivation()->getType()
                       : "Linear") << "\n";
    }

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_OUTPUTS_WIDTH*" << prefix
           << "_OUTPUTS_HEIGHT)\n"
              "#define " << prefix << "_CHANNELS_SIZE (" << prefix
           << "_NB_CHANNELS*" << prefix << "_CHANNELS_WIDTH*" << prefix
           << "_CHANNELS_HEIGHT)\n"
              "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix
           << "_OUTPUTS_SIZE, " << prefix << "_CHANNELS_SIZE))\n\n";
}

void N2D2::CPP_ConvCellExport::generateHeaderFreeParameters(ConvCell& cell,
                                                            std::ofstream
                                                            & header)
{
    generateHeaderBias(cell, header);
    generateHeaderWeights(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderBias(ConvCell& cell,
                                                  std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderBiasVariable(ConvCell& cell,
                                                          std::ofstream& header)
{
    header << "const std::vector<WDATA_T> "
        << Utils::CIdentifier(cell.getName()) << "_biases = ";
}

void N2D2::CPP_ConvCellExport::generateHeaderBiasValues(ConvCell& cell,
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

void N2D2::CPP_ConvCellExport::generateHeaderWeights(ConvCell& cell,
                                                     std::ofstream& header)
{
    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (!cell.isConnection(channel, output))
                continue;

            generateHeaderKernelWeightsVariable(cell, header, output, channel);
            generateHeaderKernelWeightsValues(
                cell, header, output, channel);
        }
    }

    generateHeaderWeightsVariable(cell, header);
    generateHeaderWeightsValues(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderKernelWeightsVariable(
    ConvCell& cell,
    std::ofstream& header,
    unsigned int output,
    unsigned int channel)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "const std::vector<std::vector<WDATA_T> > "
        << identifier
        << "_weights_" << output << "_" << channel << " = ";
}


void
N2D2::CPP_ConvCellExport::generateHeaderKernelWeightsValues(ConvCell& cell,
                                                          std::ofstream& header,
                                                          unsigned int output,
                                                          unsigned int channel)
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                        cell.getName()));

    header << "{\n";

    for (unsigned int sy = 0; sy < cell.getKernelHeight(); ++sy) {
        if (sy > 0)
            header << ",\n";

        header << "    {";

        for (unsigned int sx = 0; sx < cell.getKernelWidth(); ++sx) {
            if (sx > 0)
                header << ", ";

            CellExport::generateFreeParameter(
                cell, cell.getWeight(output, channel, sx, sy), header);
        }

        header << "}";
    }

    header << "};\n";
}

void N2D2::CPP_ConvCellExport::generateHeaderWeightsVariable(ConvCell& cell,
                                                             std::ofstream
                                                             & header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "typedef const std::vector<std::vector<WDATA_T> > " << prefix
           << "_KERNEL_T;\n";
    header << "const std::vector<std::vector<" << prefix << "_KERNEL_T*> > "
           << identifier << "_weights = ";
}

void N2D2::CPP_ConvCellExport::generateHeaderWeightsValues(ConvCell& cell,
                                                         std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "{\n";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ",\n";

        header << "    {";

        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (channel > 0)
                header << ", ";

            if (!cell.isConnection(channel, output))
                header << "NULL";
            else
                header << "&" << identifier
                    << "_weights_" << output << "_" << channel;
        }

        header << "}";
    }

    header << "};\n\n";
}
