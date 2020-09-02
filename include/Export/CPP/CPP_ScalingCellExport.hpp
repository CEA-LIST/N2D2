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

#ifndef N2D2_CPP_SCALING_CELL_EXPORT_H
#define N2D2_CPP_SCALING_CELL_EXPORT_H

#include "Export/CPP/CPP_CellExport.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/ScalingCellExport.hpp"


namespace N2D2 {

class ScalingCell;

class CPP_ScalingCellExport : public CPP_CellExport {
public:
    static void generate(const ScalingCell& cell, const std::string& dirName);

    static std::unique_ptr<CPP_ScalingCellExport> getInstance(Cell& cell);
    void generateCallCode(const DeepNet& deepNet,
                            const Cell& cell,
                            std::stringstream& includes,
                            std::stringstream& buffers, 
                            std::stringstream& functionCalls);

private:
    static void generateHeaderConstants(const ScalingCell& cell, std::ofstream& header);
};
}

#endif // N2D2_CPP_SCALING_CELL_EXPORT_H
