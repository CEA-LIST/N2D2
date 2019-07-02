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

#include "Export/C_HLS/C_HLS_PoolCellExport.hpp"

N2D2::Registrar<N2D2::PoolCellExport>
N2D2::C_HLS_PoolCellExport::mRegistrar("C_HLS",
                                       N2D2::C_HLS_PoolCellExport::generate);

N2D2::Registrar<N2D2::C_HLS_CellExport>
N2D2::C_HLS_PoolCellExport::mRegistrarType(
    PoolCell::Type, N2D2::C_HLS_PoolCellExport::getInstance);

void N2D2::C_HLS_PoolCellExport::generate(PoolCell& cell,
                                          const std::string& dirName)
{
    C_PoolCellExport::generate(cell, dirName);
}

std::unique_ptr<N2D2::C_HLS_PoolCellExport>
N2D2::C_HLS_PoolCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<C_HLS_PoolCellExport>(new C_HLS_PoolCellExport);
}

N2D2::C_HLS_CellExport::TclDirectives
N2D2::C_HLS_PoolCellExport::getTclDirectives(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& /*parentCells*/,
    bool isUnsigned)
{
    std::stringstream funcName;
    funcName << Utils::CIdentifier(cell.getName()) << "_"
             << ((isUnsigned) ? "u" : "") << "propagate";

    if (cell.isUnitMap())
        funcName << "_unitmap";

    const PoolCell* poolCell = dynamic_cast<PoolCell*>(&cell);

    if (poolCell->getPooling() == PoolCell::Average)
        funcName << "_average";

    return TclDirectives(funcName.str(),
                         (cell.isUnitMap()) ? "Pool_UnitMap" : "Pool",
                         cell.getNbChannels(),
                         cell.getChannelsWidth(),
                         cell.getChannelsHeight(),
                         poolCell->getPoolWidth(),
                         poolCell->getPoolHeight());
}

void N2D2::C_HLS_PoolCellExport::generateCellPrototype(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& /*parentCells*/,
    const std::string& outputSizeName,
    std::ofstream& prog,
    bool isUnsigned)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "POOLCELL_" << ((isUnsigned) ? "U" : "") << "PROPAGATE("
        << identifier << ", "
        << prefix << "_NB_CHANNELS, "
        << prefix << "_CHANNELS_HEIGHT, "
        << prefix << "_CHANNELS_WIDTH, "
        << outputSizeName << ", "
        << prefix << "_OUTPUTS_HEIGHT, "
        << prefix << "_OUTPUTS_WIDTH)\n";
}
