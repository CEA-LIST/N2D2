/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_CPP_TensorRT_CELLEXPORT_H
#define N2D2_CPP_TensorRT_CELLEXPORT_H

#include "Export/CPP/CPP_CellExport.hpp"
#include "Cell/Cell.hpp"
#include "Export/DeepNetExport.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {
/**
 * Virtual base class for methods commun to every cell type for the CPP_TensorRT
 * export
 * ANY CELL, CPP_TensorRT EXPORT
*/
class CPP_TensorRT_CellExport {
public:
    typedef std::function
        <std::unique_ptr<CPP_TensorRT_CellExport>(Cell&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static void generateHeaderIncludes(Cell& cell, std::ofstream& header);
    static void generateHeaderTensorRTConstants(Cell& cell,
                                                std::ofstream& header);
    static void generateAddConcat(Cell& cell,
                               std::vector<std::string>& parentsName,
                               std::ofstream& prog);

    static void generateTensor(Cell& cell,
                               std::vector<std::string>& parentsName,
                               std::ofstream& prog);

    static void generateProgramAddActivation(Cell& cell, std::ofstream& prog);

    inline static std::unique_ptr<CPP_TensorRT_CellExport> getInstance(Cell& cell);

    // Commun methods for all cells

    virtual void generateCellProgramDescriptors(Cell& cell, std::ofstream& prog) = 0;

    virtual void generateCellProgramAllocateMemory(unsigned int targetIdx, std::ofstream& prog) = 0;

    virtual void generateCellProgramInstanciateLayer(Cell& cell,
                                                    std::vector<std::string>& parentsName,
                                                    std::ofstream& prog) = 0;

    virtual void generateCellProgramInstanciateOutput(Cell& cell,
                                                       unsigned int targetIdx,
                                                       std::ofstream& prog) = 0;

};
}

std::unique_ptr<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_CellExport::getInstance(Cell& cell)
{
    return Registrar<CPP_TensorRT_CellExport>::create(cell.getType())(cell);
}

#endif // N2D2_CPP_TensorRT_CELLEXPORT_H
