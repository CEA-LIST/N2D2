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

#ifndef N2D2_C_HLS_CELLEXPORT_H
#define N2D2_C_HLS_CELLEXPORT_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Cell/Cell.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {
/**
 * Virtual base class for methods commun to every cell type for the C_HLS export
 * ANY CELL, C_HLS EXPORT
*/
class C_HLS_CellExport {
public:
    typedef std::function
        <std::unique_ptr<C_HLS_CellExport>(Cell&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    inline static std::unique_ptr<C_HLS_CellExport> getInstance(Cell& cell);

    // Commun methods for all cells
    struct TclDirectives {
        TclDirectives(std::string funcName_,
                      std::string typeName_,
                      unsigned int nbChannels_,
                      unsigned int channelsWidth_,
                      unsigned int channelsHeight_,
                      unsigned int kernelWidth_,
                      unsigned int kernelHeight_)
            : funcName(funcName_),
              typeName(typeName_),
              nbChannels(nbChannels_),
              channelsWidth(channelsWidth_),
              channelsHeight(channelsHeight_),
              kernelWidth(kernelWidth_),
              kernelHeight(kernelHeight_)
        {
        }

        std::string funcName;
        std::string typeName;
        unsigned int nbChannels;
        unsigned int channelsWidth;
        unsigned int channelsHeight;
        unsigned int kernelWidth;
        unsigned int kernelHeight;
    };

    virtual TclDirectives getTclDirectives(
        Cell& cell,
        const std::vector<std::shared_ptr<Cell> >& parentCells,
        bool isUnsigned = false) = 0;
    virtual void generateCellPrototype(Cell& cell,
                                       const std::vector
                                       <std::shared_ptr<Cell> >& parentCells,
                                       const std::string& outputSizeName,
                                       std::ofstream& prog,
                                       bool isUnsigned = false) = 0;
};
}

std::unique_ptr<N2D2::C_HLS_CellExport>
N2D2::C_HLS_CellExport::getInstance(Cell& cell)
{
    return Registrar<C_HLS_CellExport>::create(cell.getType())(cell);
}

#endif // N2D2_C_HLS_CELLEXPORT_H
