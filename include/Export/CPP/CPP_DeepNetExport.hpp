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

#ifndef N2D2_CPP_DEEPNETEXPORT_H
#define N2D2_CPP_DEEPNETEXPORT_H

#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/MemoryManager.hpp"

namespace N2D2 {
/**
 * Class for methods of DeepNet for all CPP exports types
 * DeepNetExport, CPP_EXPORT
**/
class CPP_DeepNetExport : public DeepNetExport {
public:
    static void generate(DeepNet& deepNet, const std::string& dirName);
    static void generateParamsHeader(const std::string& fileName);
    static void generateEnvironmentHeader(DeepNet& deepNet,
                                          const std::string& fileName);
    static void generateHeaderBegin(DeepNet& deepNet,
                                    std::ofstream& header,
                                    const std::string& fileName);
    static void generateHeaderIncludes(DeepNet& deepNet,
                                       const std::string typeStr,
                                       std::ofstream& header);
    static void generateHeaderEnd(DeepNet& deepNet, std::ofstream& header);

    static void generateHeaderUtils(std::ofstream& header);
    static void generateProgramUtils(std::ofstream& prog);

    static void generateMemoryInfoHeader(const DeepNet& deepNet, 
                                         const std::string& filePath, 
                                         const MemoryManager& memManager,
                                         int memoryAlignment);
    static void generateNetworkPropagateFile(const DeepNet& deepNet, 
                                             const std::string& filePath);
    static void printStats(const DeepNet& deepNet, 
                           const MemoryManager& memManager);

    static MemoryManager generateMemory(DeepNet& deepNet,
                                        bool wrapAroundBuffer,
                                        bool noBranchConcatOpt,
                                        bool includeInputInBuffer,
                                        int memoryAlignment);
    static void addBranchesCells(DeepNet& deepNet);

    static void generateEnvironmentQATHeader(DeepNet& deepNet,
                                             const std::string& fileName);
    static void generateNetworkPropagateQATFile(const DeepNet& deepNet, 
                                                const std::string& filePath);
    static MemoryManager generateQATMemory(DeepNet& deepNet,
                                           bool wrapAroundBuffer,
                                           bool noBranchConcatOpt,
                                           bool includeInputInBuffer,
                                           int memoryAlignment);

private:
    static std::string getCellModelType(const Cell& cell);

    static Registrar<DeepNetExport> mRegistrar;
};
}

#endif // N2D2_CPP_DEEPNETEXPORT_H


