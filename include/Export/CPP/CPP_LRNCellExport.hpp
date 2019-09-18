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

#ifndef N2D2_CPP_LRNCELLEXPORT_H
#define N2D2_CPP_LRNCELLEXPORT_H

#include "Export/CPP/CPP_CellExport.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/LRNCellExport.hpp"

namespace N2D2 {
/**
 * Class for methods of LRN for all CPP exports type
 * LRNCell, CPP_EXPORT
**/
class CPP_LRNCellExport : public LRNCellExport {
public:
    static void generate(const LRNCell& cell, const std::string& dirName);
    static void generateHeaderConstants(const LRNCell& cell, std::ofstream& header);
    static void generateHeaderFreeParameters(const LRNCell& cell, std::ofstream& header);

private:
    static Registrar<LRNCellExport> mRegistrar;
};
}

#endif // N2D2_CPP_LRNCELLEXPORT_H

