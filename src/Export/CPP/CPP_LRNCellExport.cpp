
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

#include "Export/CPP/CPP_LRNCellExport.hpp"

N2D2::Registrar<N2D2::LRNCellExport>
N2D2::CPP_LRNCellExport::mRegistrar("CPP", N2D2::CPP_LRNCellExport::generate);

void N2D2::CPP_LRNCellExport::generate(const LRNCell& cell, const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_LRNCellExport::generateHeaderConstants(const LRNCell& cell, std::ofstream& header)
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                    cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n\n";

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*" 
                                                        << prefix << "_OUTPUTS_WIDTH*" 
                                                        << prefix << "_OUTPUTS_HEIGHT)\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix << "_NB_CHANNELS*" 
                                                         << prefix << "_CHANNELS_WIDTH*" 
                                                         << prefix << "_CHANNELS_HEIGHT)\n\n";
}

void N2D2::CPP_LRNCellExport::generateHeaderFreeParameters(const LRNCell& cell,
                                                           std::ofstream& header)
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                        cell.getName()));
    header << "static int " << prefix
           << "_WINDOWS = " << (int) cell.getLRNn() << ";\n";

    header << "static WDATA_T " << prefix
           << "_ALPHA = ";
    CellExport::generateFreeParameter(cell.getLRNalpha(), header);
    header << ";\n";

    header << "static WDATA_T " << prefix
           << "_BETA = ";
    CellExport::generateFreeParameter(cell.getLRNbeta(), header);
    header << ";\n";

    header << "static WDATA_T " << prefix
           << "_K = ";
    CellExport::generateFreeParameter(cell.getLRNk(), header);
    header << ";\n";

}

