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
    Utils::createDirectories(dirName + "/include");

    const std::string fileName = dirName + "/include/" + cell.getName()
                                 + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    C_CellExport::generateHeaderBegin(cell, header);
    C_PoolCellExport::generateHeaderIncludes(cell, header);
    C_PoolCellExport::generateHeaderConstants(cell, header);
    generateHeaderConnections(cell, header);
    C_PoolCellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_PoolCellExport::generateHeaderConnections(PoolCell& cell,
                                                         std::ofstream& header)
{
    generateHeaderConnectionsVariable(cell, header);
    C_PoolCellExport::generateHeaderConnectionsValues(cell, header);
}

void N2D2::CPP_PoolCellExport::generateHeaderConnectionsVariable(PoolCell& cell,
                                                                 std::ofstream
                                                                 & header)
{
    header << "const std::vector<std::vector<char> > " << cell.getName()
           << "_mapping = ";
}
