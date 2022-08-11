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

#ifndef N2D2_SCALING_CELL_EXPORT_H
#define N2D2_SCALING_CELL_EXPORT_H

#include <functional>
#include <string>

#include "Export/CellExport.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {

class Cell;
class ScalingCell;

class ScalingCellExport: public CellExport {
public:
    using RegistryCreate_T = std::function<void(ScalingCell& cell, const std::string& dirName)>;

    static RegistryMap_T& registry();

    static void generate(Cell& cell, const std::string& dirName, const std::string& type);
    static bool isExportableTo(const std::string& type){
        return Registrar<ScalingCellExport>::exists(type);
    };
};
}

#endif // N2D2_SCALING_CELL_EXPORT_H


