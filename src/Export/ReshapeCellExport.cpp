/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include <string>
#include "Cell/ReshapeCell.hpp"
#include "Export/ReshapeCellExport.hpp"
#include "utils/Registrar.hpp"

static const N2D2::Registrar<N2D2::CellExport> registrar(N2D2::ReshapeCell::Type, 
                                                         N2D2::ReshapeCellExport::generate);

N2D2::RegistryMap_T& N2D2::ReshapeCellExport::registry() {
    static RegistryMap_T rMap;
    return rMap;
}

void N2D2::ReshapeCellExport::generate(Cell& cell, const std::string& dirName,
                                       const std::string& type)
{
    if (Registrar<ReshapeCellExport>::exists(type)) {
        std::cout << "-> Generating cell " << cell.getName() << std::endl;
        Registrar<ReshapeCellExport>::create(type)(dynamic_cast<ReshapeCell&>(cell), dirName);
    }
}



