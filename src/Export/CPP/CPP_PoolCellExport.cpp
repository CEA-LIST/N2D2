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

#include "Export/CPP/CPP_PoolCellExport.hpp"

N2D2::Registrar<N2D2::PoolCellExport>
N2D2::CPP_PoolCellExport::mRegistrar("CPP", N2D2::CPP_PoolCellExport::generate);

void N2D2::CPP_PoolCellExport::generate(PoolCell& cell,
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
    generateHeaderConnections(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_PoolCellExport::generateHeaderConstants(PoolCell& cell,
                                                            std::ofstream
                                                            & header)
{
    // Constants
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
                                          "#define " << prefix << "_POOL_WIDTH "
           << cell.getPoolWidth() << "\n"
                                     "#define " << prefix << "_POOL_HEIGHT "
           << cell.getPoolHeight() << "\n"
                                      "#define " << prefix << "_PADDING_X "
           << cell.getPaddingX() << "\n"
                                      "#define " << prefix << "_PADDING_Y "
           << cell.getPaddingY() << "\n"
                                      "#define " << prefix << "_STRIDE_X "
           << cell.getStrideX() << "\n"
                                   "#define " << prefix << "_STRIDE_Y "
           << cell.getStrideY() << "\n"
                                   "#define " << prefix << "_POOLING "
           << cell.getPooling() << "\n\n";

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

void N2D2::CPP_PoolCellExport::generateHeaderConnections(PoolCell& cell,
                                                         std::ofstream& header)
{
    generateHeaderConnectionsVariable(cell, header);
    generateHeaderConnectionsValues(cell, header);
}

void N2D2::CPP_PoolCellExport::generateHeaderConnectionsVariable(PoolCell& cell,
                                                                 std::ofstream
                                                                 & header)
{
    header << "const std::vector<std::vector<char> > "
        << Utils::CIdentifier(cell.getName()) << "_mapping = ";
}

void N2D2::CPP_PoolCellExport::generateHeaderConnectionsValues(PoolCell& cell,
                                                             std::ofstream
                                                             & header)
{
    header << "{";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ",\n";

        header << "    {";

        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (channel > 0)
                header << ", ";

            if (!cell.isConnection(channel, output))
                header << "0";
            else
                header << "1";
        }

        header << "}";
    }

    header << "};\n\n";
}
