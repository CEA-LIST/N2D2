/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_CPP_DECONVCELLEXPORT_H
#define N2D2_CPP_DECONVCELLEXPORT_H

#include "Export/CPP/CPP_CellExport.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/DeconvCellExport.hpp"

namespace N2D2 {
/**
 * Class for methods of DeconvCell for all CPP exports type
 * DeconvCell, CPP_EXPORT
**/

class CPP_DeconvCellExport : public DeconvCellExport {
public:
    static void generate(DeconvCell& cell, const std::string& dirName);
    static void generateHeaderConstants(DeconvCell& cell,
                                        std::ofstream& header);
    static void generateHeaderFreeParameters(DeconvCell& /*cell*/,
                                             std::ofstream& /*header*/) {};

private:
    static Registrar<DeconvCellExport> mRegistrar;
};
}

#endif // N2D2_CPP_DECONVCELLEXPORT_H

