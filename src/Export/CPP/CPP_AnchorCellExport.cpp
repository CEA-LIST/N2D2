
/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Export/CPP/CPP_AnchorCellExport.hpp"

N2D2::Registrar<N2D2::AnchorCellExport>
N2D2::CPP_AnchorCellExport::mRegistrar(
    "CPP", N2D2::CPP_AnchorCellExport::generate);

void N2D2::CPP_AnchorCellExport::generate(AnchorCell& cell,
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
    generateHeaderAnchorsParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_AnchorCellExport::generateHeaderConstants(AnchorCell& cell,
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
           << cell.getChannelsHeight() << "\n\n";

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_OUTPUTS_WIDTH*" << prefix
           << "_OUTPUTS_HEIGHT)\n"
              "#define " << prefix << "_CHANNELS_SIZE (" << prefix
           << "_NB_CHANNELS*" << prefix << "_CHANNELS_WIDTH*" << prefix
           << "_CHANNELS_HEIGHT)\n\n";

    header << "#define " << prefix << "_FM_WIDTH "
                         << cell.getFeatureMapWidth() << "\n"
             << "#define " << prefix << "_FM_HEIGHT " << cell.getFeatureMapHeight() << "\n";

    header << "#define " << prefix << "_FLIP " << cell.isFlip() << "\n"
           << "#define " << prefix << "_SCORE_CLS "
                         << cell.getScoreCls() << "\n";

}

void N2D2::CPP_AnchorCellExport
        ::generateHeaderAnchorsParameters(AnchorCell& cell,
                                          std::ofstream& header)
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                    cell.getName()));
    const std::string identifier = Utils::CIdentifier( cell.getName());

    header << "#define " << prefix << "_NB_ANCHORS "
                         << cell.getNbAnchors() << "\n";

    header << "static const WDATA_T " << identifier << "_anchors["
           << prefix << "_NB_ANCHORS][4] = {";

    for(unsigned int i = 0; i < (unsigned int) cell.getNbAnchors(); ++i)
    {
        const std::vector<Float_T> anchor_param = cell.getAnchor(i);

        if(i > 0)
            header << ",\n";

        header << "{";

        for(unsigned int paramIdx = 0; paramIdx < anchor_param.size();
            ++paramIdx)
        {
            header << std::setprecision(10) << anchor_param[paramIdx];
            if(paramIdx < anchor_param.size() - 1)
                header << ", ";
        }

        header << "}";
    }

    header << "};\n";
}
