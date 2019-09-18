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

#include "Export/C/C_FMPCellExport.hpp"

N2D2::Registrar<N2D2::FMPCellExport>
N2D2::C_FMPCellExport::mRegistrar("C", N2D2::C_FMPCellExport::generate);

N2D2::Registrar<N2D2::C_CellExport>
N2D2::C_FMPCellExport::mRegistrarType(FMPCell::Type,
                                      N2D2::C_FMPCellExport::getInstance);

void N2D2::C_FMPCellExport::generate(const FMPCell& cell, const std::string& dirName)
{
    Utils::createDirectories(dirName + "/include");

    const std::string fileName = dirName + "/include/"
        + Utils::CIdentifier(cell.getName()) + ".h";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    C_CellExport::generateHeaderBegin(cell, header);
    C_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    generateHeaderConnections(cell, header);
    generateHeaderGrid(cell, header);
    C_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::C_FMPCellExport::generateHeaderConstants(const FMPCell& cell,
                                                    std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n"
           << "#define " << prefix << "_OVERLAPPING " << cell.getParameter<bool>("Overlapping") << "\n"
           << "#define " << prefix << "_PSEUDO_RANDOM " << cell.getParameter<bool>("PseudoRandom") << "\n";

    C_CellExport::generateActivation(cell, header);

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*" 
                                                        << prefix << "_OUTPUTS_WIDTH*" 
                                                        << prefix << "_OUTPUTS_HEIGHT)\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix << "_NB_CHANNELS*" 
                                                         << prefix << "_CHANNELS_WIDTH*" 
                                                         << prefix << "_CHANNELS_HEIGHT)\n"
           << "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix << "_OUTPUTS_SIZE, " 
                                                           << prefix << "_CHANNELS_SIZE))\n\n";
}

void N2D2::C_FMPCellExport::generateHeaderConnections(const FMPCell& cell,
                                                      std::ofstream& header)
{
    if (!cell.isUnitMap()) {
        const std::string identifier = Utils::CIdentifier(cell.getName());
        const std::string prefix = Utils::upperCase(identifier);

        header << "#define " << prefix << "_MAPPING_SIZE (" << prefix
               << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n"
               << "static char " << identifier << "_mapping_flatten["
               << prefix << "_MAPPING_SIZE] = {\n";

        for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
            for (std::size_t channel = 0; channel < cell.getNbChannels();
                 ++channel) {
                if (output > 0 || channel > 0)
                    header << ", ";

                if (!cell.isConnection(channel, output))
                    header << "0";
                else
                    header << "1";
            }
        }

        header << "};\n\n";
    }
}

void N2D2::C_FMPCellExport::generateHeaderGrid(const FMPCell& cell,
                                               std::ofstream& header)
{

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    const unsigned int sizeOutX = cell.getOutputsWidth();
    const unsigned int sizeInX = cell.getChannelsWidth();

    const unsigned int sizeOutY = cell.getOutputsHeight();
    const unsigned int sizeInY = cell.getChannelsHeight();

    const double scalingRatioX = sizeInX / (double)sizeOutX;
    const double scalingRatioY = sizeInY / (double)sizeOutY;

    const double uX = Random::randUniform(0.0, 1.0, Random::OpenInterval);
    const double uY = Random::randUniform(0.0, 1.0, Random::OpenInterval);

    if (cell.getParameter<bool>("PseudoRandom")) {

        header << "#define " << prefix << "_GRIDX_SIZE (" << prefix
               << "_OUTPUTS_WIDTH)\n"
               << "static unsigned int " << identifier << "_gridx_flatten["
               << prefix << "_OUTPUTS_WIDTH] = {\n";

        for (unsigned int i = 0; i < sizeOutX; ++i) {
            if (i > 0)
                header << ", ";

            header << (unsigned int)std::ceil(scalingRatioX * (i + uX));
        }
        header << "};\n\n";

        header << "#define " << prefix << "_GRIDY_SIZE (" << prefix
               << "_OUTPUTS_HEIGHT)\n"
               << "static unsigned int " << identifier << "_gridy_flatten["
               << prefix << "_OUTPUTS_HEIGHT] = {\n";

        for (unsigned int i = 0; i < sizeOutY; ++i) {
            if (i > 0)
                header << ", ";

            header << (unsigned int)std::ceil(scalingRatioY * (i + uY));
        }
        header << "};\n\n";
    } else {
        const unsigned int nb2X = sizeInX - sizeOutX;
        const unsigned int nb2Y = sizeInY - sizeOutY;

        // const unsigned int nb1 = 2*sizeOut - sizeIn;
        // assert(nb1 + nb2 == sizeOut);
        std::vector<unsigned int> gridX;
        std::vector<unsigned int> gridY;

        std::fill(gridX.begin(), gridX.begin() + nb2X, 2);
        std::fill(gridY.begin(), gridY.begin() + nb2Y, 2);

        std::fill(gridX.begin() + nb2X, gridX.end(), 1);
        std::fill(gridY.begin() + nb2Y, gridY.end(), 1);

        // Random shuffle
        for (int i = gridX.size() - 1; i > 0; --i)
            std::swap(gridX[i], gridX[Random::randUniform(0, i)]);
        // Random shuffle
        for (int i = gridY.size() - 1; i > 0; --i)
            std::swap(gridY[i], gridY[Random::randUniform(0, i)]);

        for (unsigned int i = 1; i < gridX.size(); ++i)
            gridX[i] += gridX[i - 1];

        for (unsigned int i = 1; i < gridY.size(); ++i)
            gridY[i] += gridY[i - 1];

        header << "#define " << prefix << "_GRIDX_SIZE (" << prefix
               << "_OUTPUTS_WIDTH)\n"
               << "static unsigned int " << identifier << "_gridx_flatten["
               << prefix << "_OUTPUTS_WIDTH] = {\n";

        for (unsigned int i = 1; i < gridX.size(); ++i) {
            if (i > 0)
                header << ", ";
            header << gridX[i];
        }
        header << "};\n\n";

        header << "#define " << prefix << "_GRIDY_SIZE (" << prefix
               << "_OUTPUTS_HEIGHT)\n"
               << "static unsigned int " << identifier << "_gridy_flatten["
               << prefix << "_OUTPUTS_HEIGHT] = {\n";

        for (unsigned int i = 1; i < gridY.size(); ++i) {
            if (i > 0)
                header << ", ";
            header << gridY[i];
        }
        header << "};\n\n";
    }
}

std::unique_ptr<N2D2::C_FMPCellExport>
N2D2::C_FMPCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<C_FMPCellExport>(new C_FMPCellExport);
}

void N2D2::C_FMPCellExport::generateCellData(Cell& cell,
                                             const std::string& outputName,
                                             const std::string& outputSizeName,
                                             std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "static DATA_T " << outputName << "[" << outputSizeName << "]["
         << prefix << "_OUTPUTS_HEIGHT][" << prefix << "_OUTPUTS_WIDTH];\n";
}

void N2D2::C_FMPCellExport::generateCellFunction(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& /*parentCells*/,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& outputSizeName,
    std::ofstream& prog,
    bool isUnsigned,
    const std::string& funcProto,
    const std::string& /*memProto*/,
    bool /*memCompact*/)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto = (funcProto.empty()) ? "fmpcell" : funcProto;

    prog << "    " << proto << "_" << ((isUnsigned) ? "u" : "") << "propagate("
        << prefix << "_NB_CHANNELS, "
        << prefix << "_CHANNELS_HEIGHT, "
        << prefix << "_CHANNELS_WIDTH, "
        << prefix << "_OVERLAPPING, "
        << inputName << ", "
        << outputSizeName << ", "
        << prefix << "_OUTPUTS_HEIGHT, "
        << prefix << "_OUTPUTS_WIDTH, "
        << prefix << "_NB_OUTPUTS, "
        << prefix << "_OUTPUT_OFFSET, "
        << outputName << ", "
        << identifier << "_gridx_flatten, "
        << identifier << "_gridy_flatten, "
        << prefix << "_ACTIVATION);\n";
}

void N2D2::C_FMPCellExport::generateOutputFunction(Cell& cell,
                                                   const std::string& inputName,
                                                   const std::string
                                                   & outputName,
                                                   std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    if ((cell.getOutputsWidth() == 1) && (cell.getOutputsHeight() == 1)) {
        prog << "\n"
                "    output_max(" << prefix << "_NB_OUTPUTS, " << inputName
             << ", " << outputName << ");\n";
    } else {
        prog << "\n"
                "    spatial_output_max(" << prefix << "_NB_OUTPUTS, " << prefix
             << "_OUTPUTS_HEIGHT, " << prefix << "_OUTPUTS_WIDTH, " << inputName
             << ", " << outputName << ");\n";
    }
}
