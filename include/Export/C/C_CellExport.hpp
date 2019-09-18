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

#ifndef N2D2_C_CELLEXPORT_H
#define N2D2_C_CELLEXPORT_H

#include <fstream>
#include <functional>
#include <memory>
#include <string>

#include "Cell/Cell.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {
/**
 * Virtual base class for methods commun to every cell type for the C export
 * ANY CELL, C EXPORT
*/
class C_CellExport {
public:
    typedef std::function
        <std::unique_ptr<C_CellExport>(Cell&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static void generateHeaderBegin(const Cell& cell, std::ofstream& header);
    static void generateHeaderIncludes(const Cell& cell, std::ofstream& header);
    static void generateHeaderEnd(const Cell& cell, std::ofstream& header);

    static void generateActivation(const Cell& cell, std::ofstream& header);
    static void generateActivationScaling(const Cell& cell, std::ofstream& header);

    inline static std::unique_ptr<C_CellExport> getInstance(Cell& cell);

    // Commun methods for all cells
    virtual void generateCellData(Cell& cell,
                                  const std::string& outputName,
                                  const std::string& outputSizeName,
                                  std::ofstream& prog) = 0;
    virtual void generateCellFunction(Cell& cell,
                                      const std::vector
                                      <std::shared_ptr<Cell> >& parentCells,
                                      const std::string& inputName,
                                      const std::string& outputName,
                                      const std::string& outputSizeName,
                                      std::ofstream& prog,
                                      bool isUnsigned = false,
                                      const std::string& funcProto = "",
                                      const std::string& memProto = "",
                                      bool memCompact = false) = 0;
    virtual void generateOutputFunction(Cell& cell,
                                        const std::string& inputName,
                                        const std::string& outputName,
                                        std::ofstream& prog) = 0;
};
}

std::unique_ptr<N2D2::C_CellExport> N2D2::C_CellExport::getInstance(Cell& cell)
{
    return Registrar<C_CellExport>::create(cell.getType())(cell);
}

#endif // N2D2_C_CELLEXPORT_H
