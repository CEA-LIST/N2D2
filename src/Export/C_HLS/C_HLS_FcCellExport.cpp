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

#include "Export/C_HLS/C_HLS_FcCellExport.hpp"
#include "Cell/ConvCell.hpp"
#include "Cell/PoolCell.hpp"

N2D2::Registrar<N2D2::FcCellExport>
N2D2::C_HLS_FcCellExport::mRegistrar("C_HLS",
                                     N2D2::C_HLS_FcCellExport::generate);

N2D2::Registrar<N2D2::C_HLS_CellExport>
N2D2::C_HLS_FcCellExport::mRegistrarType(FcCell::Type,
                                         N2D2::C_HLS_FcCellExport::getInstance);

void N2D2::C_HLS_FcCellExport::generate(FcCell& cell,
                                        const std::string& dirName)
{
    C_FcCellExport::generate(cell, dirName);
}

std::unique_ptr<N2D2::C_HLS_FcCellExport>
N2D2::C_HLS_FcCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<C_HLS_FcCellExport>(new C_HLS_FcCellExport);
}

N2D2::C_HLS_CellExport::TclDirectives
N2D2::C_HLS_FcCellExport::getTclDirectives(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& parentCells,
    bool isUnsigned)
{
    std::stringstream funcName;
    std::stringstream typeName;
    funcName << Utils::CIdentifier(cell.getName()) << "_"
             << ((isUnsigned) ? "u" : "") << "propagate";
    typeName << "Fc";

    unsigned int nbChannels = cell.getNbChannels();
    unsigned int channelsWidth = cell.getChannelsWidth();
    unsigned int channelsHeight = cell.getChannelsHeight();

    if ((!parentCells[0]
         || parentCells[0]->getType()
            == ConvCell::Type /*|| parentCells[0]->getType() == Cell::Lc*/
         || parentCells[0]->getType() == PoolCell::Type)) {
        funcName << "_2d";
        typeName << "_2D";

        if (parentCells[0]) {
            nbChannels = parentCells[0]->getNbOutputs();
            channelsWidth = parentCells[0]->getOutputsWidth();
            channelsHeight = parentCells[0]->getOutputsHeight();
        }
    }

    return TclDirectives(funcName.str(),
                         typeName.str(),
                         nbChannels,
                         channelsWidth,
                         channelsHeight,
                         1,
                         1);
}

void N2D2::C_HLS_FcCellExport::generateCellPrototype(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& parentCells,
    const std::string& outputSizeName,
    std::ofstream& prog,
    bool isUnsigned)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const bool input2d
        = (!parentCells[0]
           || parentCells[0]->getType()
              == ConvCell::Type /*|| parentCells[0]->getType() == Cell::Lc*/
           || parentCells[0]->getType() == PoolCell::Type);

    prog << "FCCELL_" << ((isUnsigned) ? "U" : "") << "PROPAGATE"
        << ((input2d) ? "_2D" : "") << "("
        << identifier << ", ";

    if (input2d) {
        if (!parentCells[0]) {
            prog << "ENV_NB_OUTPUTS, "
                 << "ENV_SIZE_Y, "
                 << "ENV_SIZE_X, ";
        }
        else {
            std::stringstream channelInputName;

            for (unsigned int layer = 0; layer < parentCells.size(); ++layer)
                channelInputName << (*parentCells[layer]).getName() << "_";

            const std::string prefixParent
                = Utils::upperCase(parentCells[0]->getName());

            prog << Utils::upperCase(Utils::CIdentifier(channelInputName.str()))
                 << "NB_OUTPUTS, "
                 << prefixParent << "_OUTPUTS_HEIGHT, "
                 << prefixParent << "_OUTPUTS_WIDTH, ";
        }
    }
    else
        prog << prefix << "_NB_CHANNELS, ";

    prog << outputSizeName << ", ";

    if (input2d)
        prog << prefix << "_NB_CHANNELS, ";

    prog << prefix << "_WEIGHTS_SIZE)\n";
}
