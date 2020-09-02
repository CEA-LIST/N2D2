/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#include "Export/CPP/CPP_DeconvCellExport.hpp"

N2D2::Registrar<N2D2::DeconvCellExport>
N2D2::CPP_DeconvCellExport::mRegistrar(
    "CPP", N2D2::CPP_DeconvCellExport::generate);

void N2D2::CPP_DeconvCellExport::generate(DeconvCell& cell,
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
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_DeconvCellExport::generateHeaderConstants(DeconvCell& cell,
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
           << "#define " << prefix
           << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix
           << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight()
           << "\n"
              "#define " << prefix << "_KERNEL_WIDTH " << cell.getKernelWidth()
           << "\n"
              "#define " << prefix << "_KERNEL_HEIGHT "
           << cell.getKernelHeight() << "\n"
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
           << "_CHANNELS_HEIGHT)\n\n";
}
