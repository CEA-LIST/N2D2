
/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "Export/CPP/CPP_FMPCellExport.hpp"

N2D2::Registrar<N2D2::FMPCellExport>
N2D2::CPP_FMPCellExport::mRegistrar(
    "CPP", N2D2::CPP_FMPCellExport::generate);

void N2D2::CPP_FMPCellExport::generate(FMPCell& cell,
                                       const std::string& dirName)
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
    generateHeaderConnections(cell, header);
    generateHeaderGrid(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_FMPCellExport::generateHeaderConstants(FMPCell& cell,
                                                             std::ofstream
                                                             & header)
{
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
           << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth()
           << "\n"
              "#define " << prefix << "_CHANNELS_HEIGHT "
           << cell.getChannelsHeight() << "\n"
                                          "#define " << prefix
           << "_OVERLAPPING " << cell.getParameter<bool>("Overlapping")
           << "\n"
              "#define " << prefix << "_PSEUDO_RANDOM "
           << cell.getParameter<bool>("PseudoRandom") << "\n";

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

void N2D2::CPP_FMPCellExport::generateHeaderConnections(FMPCell& cell,
                                                               std::ofstream
                                                               & header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const Cell_Frame_Top* cellFrameTop = dynamic_cast<Cell_Frame_Top*>(&cell);

    if (/*cellFrame != NULL && */ !cellFrameTop->isUnitMap()) {

        const std::string prefix = Utils::upperCase(identifier);

        header << "#define " << prefix << "_MAPPING_SIZE (" << prefix
               << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n"
               << "static char " << identifier
               << "_mapping_flatten[" << prefix << "_MAPPING_SIZE] = {\n";

        for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
            for (unsigned int channel = 0; channel < cell.getNbChannels();
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

void N2D2::CPP_FMPCellExport::generateHeaderGrid(FMPCell& cell,
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
               << "static unsigned int " << identifier
               << "_gridx_flatten[" << prefix << "_OUTPUTS_WIDTH] = {\n";

        for (unsigned int i = 0; i < sizeOutX; ++i) {
            if (i > 0)
                header << ", ";

            header << (unsigned int)std::ceil(scalingRatioX * (i + uX));
        }
        header << "};\n\n";

        header << "#define " << prefix << "_GRIDY_SIZE (" << prefix
               << "_OUTPUTS_HEIGHT)\n"
               << "static unsigned int " << identifier
               << "_gridy_flatten[" << prefix << "_OUTPUTS_HEIGHT] = {\n";

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
               << "static unsigned int " << cell.getName() << "_gridx_flatten["
               << prefix << "_OUTPUTS_WIDTH] = {\n";

        for (unsigned int i = 1; i < gridX.size(); ++i) {
            if (i > 0)
                header << ", ";
            header << gridX[i];
        }
        header << "};\n\n";

        header << "#define " << prefix << "_GRIDY_SIZE (" << prefix
               << "_OUTPUTS_HEIGHT)\n"
               << "static unsigned int " << cell.getName() << "_gridy_flatten["
               << prefix << "_OUTPUTS_HEIGHT] = {\n";

        for (unsigned int i = 1; i < gridY.size(); ++i) {
            if (i > 0)
                header << ", ";
            header << gridY[i];
        }
        header << "};\n\n";
    }
}

