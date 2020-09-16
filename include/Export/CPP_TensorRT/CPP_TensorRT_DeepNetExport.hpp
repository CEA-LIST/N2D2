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
#ifndef N2D2_CPP_TensorRT_DEEPNETEXPORT_H
#define N2D2_CPP_TensorRT_DEEPNETEXPORT_H

#include "Export/CPP_TensorRT/CPP_TensorRT_StimuliProvider.hpp"
#include "Export/CPP_TensorRT/CPP_TensorRT_CellExport.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/CPP/CPP_DeepNetExport.hpp"
#include "Export/DeepNetExport.hpp"

namespace N2D2 {
/**
 * Class for methods of DeepNet for CPP_TensorRT exports types
 * DeepNetExport, CPP_TensorRT EXPORT
**/

class CPP_TensorRT_DeepNetExport : public DeepNetExport {
public:
    static void generate(DeepNet& deepNet, const std::string& dirName);
    static void generateDeepNetProgram(DeepNet& deepNet,
                                       const std::string& name,
                                       const std::string& fileName);
    static void generateStimuliCalib(DeepNet& deepNet,
                                const std::string& dirName);
    static void generateProgramBegin(DeepNet& deepNet,
                                     std::ofstream& prog);
    static void generateIncludes(DeepNet& deepNet,
                                     std::ofstream& prog);
    static void generateProgramInitNetwork(DeepNet& deepNet,
                                           const std::string& name,
                                           std::ofstream& prog);

private:
    static Registrar<DeepNetExport> mRegistrar;
};
}

#endif // N2D2_CPP_TensorRT_DEEPNETEXPORT_H
