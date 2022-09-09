
/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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

#include "Export/CPP/CPP_ActivationCellExport.hpp"

#include "Export/CPP/Cells/CPP_ConcatCell.hpp"
#include "Export/CPP/CPP_ConcatCellExport.hpp"

N2D2::Registrar<N2D2::ActivationCellExport>
N2D2::CPP_ActivationCellExport::mRegistrar(
    "CPP", N2D2::CPP_ActivationCellExport::generate);

N2D2::Registrar<N2D2::CPP_CellExport> N2D2::CPP_ActivationCellExport::mRegistrarType(
    N2D2::ActivationCell::Type, N2D2::CPP_ActivationCellExport::getInstance);

void N2D2::CPP_ActivationCellExport::generate(ActivationCell& cell,
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
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_ActivationCellExport::generateHeaderConstants(ActivationCell& cell,
                                                            std::ofstream
                                                            & header)
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                    cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs()
           << "\n"
              "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels()
           << "\n\n";

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    std::string type = (cellFrame.getActivation())
                    ? cellFrame.getActivation()->getType() : "Linear";

    if(type == "Linear"){
        //adding info from concat
        const auto parentsCells = cell.getParentsCells();
        header << "#define " << prefix << "_NB_INPUTS " << parentsCells.size() << "\n";

        header << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
            << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n"
            << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
            << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
            << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"

            << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*"
                                                            << prefix << "_OUTPUTS_WIDTH*"
                                                            << prefix << "_OUTPUTS_HEIGHT)" << "\n"
            << "#define " << prefix << "_CHANNELS_SIZE (" << prefix  << "_NB_CHANNELS*"
                                                            << prefix << "_CHANNELS_WIDTH*"
                                                            << prefix << "_CHANNELS_HEIGHT)" << "\n"
            << "\n";
    }
}

std::unique_ptr<N2D2::CPP_ActivationCellExport>
N2D2::CPP_ActivationCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_ActivationCellExport>(new CPP_ActivationCellExport);
}

void N2D2::CPP_ActivationCellExport::generateCallCode(
    const DeepNet& deepNet,
    const Cell& cell,
    std::stringstream& includes,
    std::stringstream& /*buffers*/,
    std::stringstream& functionCalls)
{
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    std::string type = (cellFrame.getActivation())
                    ? cellFrame.getActivation()->getType() : "Linear";

    if(type == "Linear"){
        std::stringstream buffers;

        std::unique_ptr<CPP_ConcatCellExport>(new CPP_ConcatCellExport)->CPP_ConcatCellExport::generateCallCode(deepNet, cell, includes, buffers, functionCalls);
    }
}
