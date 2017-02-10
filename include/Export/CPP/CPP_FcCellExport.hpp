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

#ifndef N2D2_CPP_FCCELLEXPORT_H
#define N2D2_CPP_FCCELLEXPORT_H

#include "Export/CPP/CPP_CellExport.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/FcCellExport.hpp"

namespace N2D2 {
/**
 * Class for methods of FcCell for all CPP exports type
 * FcCell, CPP EXPORT
**/
class CPP_FcCellExport : public FcCellExport {
public:
    static void generate(FcCell& cell, const std::string& dirName);
    static void generateHeaderConstants(FcCell& cell, std::ofstream& header);

    static void generateHeaderFreeParameters(FcCell& cell,
                                             std::ofstream& header);

    static void generateHeaderBias(FcCell& cell, std::ofstream& header);
    static void generateHeaderBiasVariable(FcCell& cell, std::ofstream& header);
    static void generateHeaderBiasValues(FcCell& cell,
                                         std::ofstream& header);

    static void generateHeaderWeights(FcCell& cell, std::ofstream& header);
    static void generateHeaderWeightsSparse(FcCell& cell,
                                            std::ofstream& header);
    static void generateHeaderWeightsVariable(FcCell& cell,
                                              std::ofstream& header);
    static void generateHeaderWeightsValues(FcCell& cell,
                                            std::ofstream& header);
private:
    static Registrar<FcCellExport> mRegistrar;
};
}

#endif // N2D2_CPP_FCCELLEXPORT_H

