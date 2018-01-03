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

#ifndef N2D2_DEEPNETEXPORT_H
#define N2D2_DEEPNETEXPORT_H

#include <functional>
#include <string>
#include <vector>

#include "DeepNet.hpp"
#include "Export/CellExport.hpp"
#include "N2D2.hpp"
#include "utils/Registrar.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@CPP_cuDNN_DeepNetExport@N2D2@@0U?$Registrar@VDeepNetExport@N2D2@@@2@A")
#endif

namespace N2D2 {
class DeepNetExport {
public:
    typedef std::function
        <void(DeepNet& deepNet, const std::string&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    /// If true, enable the handling of unsigned data in the export
    static bool mUnsignedData;
    /// Indicate whereas the input stimuli are unsigned or not.
    /// If mUnsignedData is false, mEnvDataUnsigned is necessary false too
    static bool mEnvDataUnsigned;
    static std::string mExportParameters;

    static void generate(DeepNet& deepNet,
                         const std::string& dirName,
                         const std::string& type);

protected:
    static std::string getLayerName(DeepNet& deepNet,
                                    const std::vector<std::string>& layer);
    static bool isSharedOutput(DeepNet& deepNet,
                               const unsigned int layerNumber,
                               const unsigned int cellNumber);
    static bool isSharedInput(DeepNet& deepNet,
                              const unsigned int layerNumber,
                              const unsigned int cellNumber);
    static std::vector<unsigned int>
    getMapLayer(DeepNet& deepNet, const unsigned int layerNumber);
    static std::string getCellInputName(DeepNet& deepNet,
                                        const unsigned int layerNumber,
                                        const unsigned int cellNumber);
    static std::string getCellOutputName(DeepNet& deepNet,
                                         const unsigned int layerNumber,
                                         const unsigned int cellNumber);
    static bool isCellUnsigned(DeepNet& deepNet, Cell& cell);
};
}

#endif // N2D2_DEEPNETEXPORT_H
