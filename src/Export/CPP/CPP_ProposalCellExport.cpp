
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

#include "Export/CPP/CPP_ProposalCellExport.hpp"

N2D2::Registrar<N2D2::ProposalCellExport>
N2D2::CPP_ProposalCellExport::mRegistrar(
    "CPP", N2D2::CPP_ProposalCellExport::generate);

void N2D2::CPP_ProposalCellExport::generate(ProposalCell& cell,
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
    generateHeaderProposalParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_ProposalCellExport::generateHeaderConstants(ProposalCell& cell,
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
           << "_CHANNELS_HEIGHT)\n"
              "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix
           << "_OUTPUTS_SIZE, " << prefix << "_CHANNELS_SIZE))\n\n";

}

void N2D2::CPP_ProposalCellExport
        ::generateHeaderProposalParameters(ProposalCell& cell,
                                          std::ofstream& header)
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                    cell.getName()));
    const std::string identifier = Utils::CIdentifier( cell.getName());

    header << std::setprecision(10)
           <<  "#define " << prefix << "_NB_PROPOSALS "
                        << cell.getNbProposals() << "\n"
           << "#define " << prefix << "_NMS_IUO_THRESHOLD "
                         << cell.getNMSParam() << "\n"
           << "#define " << prefix << "_SCORE_THRESHOLD "
                         << cell.getScoreThreshold() << "\n"
           << "#define " << prefix << "_SCORE_INDEX "
                         << cell.getScoreIndex() << "\n"
           << "#define " << prefix << "_IOU_INDEX "
                         << cell.getIoUIndex() << "\n"
           << "#define " << prefix << "_APPLY_NMS "
                         << cell.getIsNMS() << "\n"
           << "#define " << prefix << "_KEEP_MAX "
                         << cell.getKeepMax() << "\n"
           << "#define " << prefix << "_NB_CLASS "
                         << cell.getNbClass() << "\n"
           << "#define " << prefix << "_MAX_PARTS "
                         << cell.getMaxParts() << "\n"
           << "#define " << prefix << "_MAX_TEMPLATES "
                         << cell.getMaxTemplates() << "\n";


    const std::vector<double> meanFactor = cell.getMeanFactor();
    header << std::setprecision(10)
           <<  "static const WDATA_T " << prefix << "_MEANS[4] = {"
            << meanFactor[0] << ", "
            << meanFactor[1] << ", "
            << meanFactor[2] << ", "
            << meanFactor[3] << "};"
            << "\n";

    const std::vector<double> stdFactor = cell.getStdFactor();
    header << std::setprecision(10)
           <<  "static const WDATA_T  " << prefix << "_STD[4] = {"
            << stdFactor[0] << ", "
            << stdFactor[1] << ", "
            << stdFactor[2] << ", "
            << stdFactor[3] << "};"
            << "\n";

    const std::vector<unsigned int> partsPerClass = cell.getPartsPerClass();
    header << std::setprecision(10)
           <<  "static const unsigned int  " << prefix << "_PARTS[";

    if(partsPerClass.size() > 1)
        header << partsPerClass.size();
    else
        header << "1" ;

    header << "] = {";
    for(unsigned int i = 0; i < partsPerClass.size(); ++i)
    {
        header << partsPerClass[i];

        if(i < partsPerClass.size() - 1)
            header << ", ";
    }

    if (partsPerClass.size() == 0)
        header << "0";

    header << "};"
            << "\n";

    const std::vector<unsigned int> templatesPerClass = cell.getTemplatesPerClass();
    header << std::setprecision(10)
           <<  "static const unsigned int  " << prefix << "_TEMPLATES[" ;
    if(templatesPerClass.size() > 1)
        header << templatesPerClass.size();
    else
        header << "1" ;
    header << "] = {";

    for(unsigned int i = 0; i < templatesPerClass.size(); ++i)
    {
        header << templatesPerClass[i];
        if(i < templatesPerClass.size() - 1)
            header << ", ";

    }
    if (templatesPerClass.size() == 0)
        header << "0";
    header << "};"
            << "\n";
}
