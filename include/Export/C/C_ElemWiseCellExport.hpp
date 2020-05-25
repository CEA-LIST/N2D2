/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_C_ELEMWISE_CELL_EXPORT_H
#define N2D2_C_ELEMWISE_CELL_EXPORT_H

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "Export/C/C_CellExport.hpp"

namespace N2D2 {

class Cell;
class ElemWiseCell;

class C_ElemWiseCellExport: public C_CellExport {
public:
    static void generate(const ElemWiseCell& cell, const std::string& dirName);

    static std::unique_ptr<C_ElemWiseCellExport> getInstance(Cell& cell);
    void generateCellData(Cell& cell,
                          const std::string& outputName,
                          const std::string& outputSizeName,
                          std::ofstream& prog);

    void generateCellFunction(Cell& cell,
                              const std::vector
                              <std::shared_ptr<Cell> >& parentCells,
                              const std::string& inputName,
                              const std::string& outputName,
                              const std::string& outputSizeName,
                              std::ofstream& prog,
                              bool isUnsigned = false,
                              const std::string& funcProto = "",
                              const std::string& memProto = "",
                              bool memCompact = false);

    void generateOutputFunction(Cell& cell,
                                const std::string& inputName,
                                const std::string& outputName,
                                std::ofstream& prog);
};
}

#endif