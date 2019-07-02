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

#ifndef N2D2_C_HLS_FCCELLEXPORT_H
#define N2D2_C_HLS_FCCELLEXPORT_H

#include "Export/C/C_FcCellExport.hpp"
#include "Export/C_HLS/C_HLS_CellExport.hpp"
#include "Export/FcCellExport.hpp"

namespace N2D2 {
/**
 * Class for methods for the FcCell type for the C_HLS export
 * FcCell, C_HLS EXPORT
*/
class C_HLS_FcCellExport : public FcCellExport, public C_HLS_CellExport {
public:
    static void generate(FcCell& cell, const std::string& dirName);

    static std::unique_ptr<C_HLS_FcCellExport> getInstance(Cell& cell);
    TclDirectives getTclDirectives(Cell& cell,
                                   const std::vector
                                   <std::shared_ptr<Cell> >& parentCells,
                                   bool isUnsigned = false);
    void generateCellPrototype(Cell& cell,
                               const std::vector
                               <std::shared_ptr<Cell> >& parentCells,
                               const std::string& outputSizeName,
                               std::ofstream& prog,
                               bool isUnsigned = false);

private:
    static Registrar<FcCellExport> mRegistrar;
    static Registrar<C_HLS_CellExport> mRegistrarType;
};
}

#endif // N2D2_C_HLS_FCCELLEXPORT_H
