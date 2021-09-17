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

#ifndef N2D2_CPP_CONVCELLEXPORT_H
#define N2D2_CPP_CONVCELLEXPORT_H

#include "Export/ConvCellExport.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/CPP/CPP_CellExport.hpp"

namespace N2D2 {
/**
 * Class for methods of ConvCell for all CPP exports type
 * ConvCell, CPP_EXPORT
**/

class CPP_ConvCellExport : public ConvCellExport, public CPP_CellExport {
public:
    static void generate(const ConvCell& cell, const std::string& dirName);
    static void generateHeaderFreeParameters(const ConvCell& cell,  std::ofstream& header);

    static void generateHeaderConstants(const ConvCell& cell, std::ofstream& header);

    static void generateHeaderBias(const ConvCell& cell, std::ofstream& header);
    static void generateHeaderWeights(const ConvCell& cell, std::ofstream& header);
    static void generateHeaderWeightsQAT(const ConvCell& cell, std::ofstream& header);

    static bool isDWConvolution(const Cell& cell);

    static std::unique_ptr<CPP_ConvCellExport> getInstance(Cell& cell);
    void generateCallCode(const DeepNet& deepNet,
                                 const Cell& cell, 
                                 std::stringstream& includes,
                                 std::stringstream& buffers, 
                                 std::stringstream& functionCalls);

private:
    static Registrar<ConvCellExport> mRegistrar;
    static Registrar<CPP_CellExport> mRegistrarType;
};
}

#endif // N2D2_CPP_CONVCELLEXPORT_H

