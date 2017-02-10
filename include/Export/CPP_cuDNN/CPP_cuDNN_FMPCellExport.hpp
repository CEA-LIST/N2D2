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

#ifndef N2D2_CPP_CUDNN_FMPCELLEXPORT_H
#define N2D2_CPP_CUDNN_FMPCELLEXPORT_H

#include "Export/FMPCellExport.hpp"
#include "Export/CPP/CPP_FMPCellExport.hpp"
#include "Export/CPP_cuDNN/CPP_cuDNN_CellExport.hpp"

namespace N2D2 {
/**
 * Class for methods for the FMPCell type for the CPP_cuDNN export
 * FMPCell, CPP_cuDNN EXPORT
*/
class CPP_cuDNN_FMPCellExport : public FMPCellExport,
                                public CPP_cuDNN_CellExport {
public:
    static void generate(FMPCell& cell, const std::string& dirName);

    static std::unique_ptr<CPP_cuDNN_FMPCellExport> getInstance(Cell& cell);

    void generateCellProgramDesc(Cell& cell, std::ofstream& prog);
    void generateCellProgramTensorDesc(Cell& cell, std::ofstream& prog);
    void generateCellProgramGlobalDefinition(Cell& cell, std::ofstream& prog);
    void generateCellBuffer(const std::string& bufferName, std::ofstream& prog);

    void generateCellProgramInitNetwork(Cell& cell,
                                       std::vector<std::string>& parentsName,
                                       std::ofstream& prog);
    void generateCellProgramInitBuffer(Cell& cell,
                                       const std::string& bufferName,
                                       std::ofstream& prog);
    void generateCellProgramFunction(Cell& cell,
                                     const std::string& inputName,
                                     const std::string& outputName,
                                     const std::string& output_pos,
                                     std::ofstream& prog,
                                     const std::string& funcProto = "");
    void generateCellProgramOutputFunction(Cell& cell,
                                           const std::string& outputDataName,
                                           const std::string& outputName,
                                           std::ofstream& prog);
    void generateCellProgramFree(Cell& cell,
                                 std::vector<std::string>& parentsName,
                                 std::ofstream& prog);

private:
    static Registrar<FMPCellExport> mRegistrar;
    static Registrar<CPP_cuDNN_CellExport> mRegistrarType;
};
}

#endif // N2D2_CPP_CUDNN_POOLCELLEXPORT_H
