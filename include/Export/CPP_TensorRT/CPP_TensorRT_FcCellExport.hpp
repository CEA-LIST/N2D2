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

#ifndef N2D2_CPP_TensorRT_FCCELLEXPORT_H
#define N2D2_CPP_TensorRT_FCCELLEXPORT_H

#include "Export/CPP_TensorRT/CPP_TensorRT_CellExport.hpp"
#include "Export/CPP/CPP_FcCellExport.hpp"
#include "Export/FcCellExport.hpp"

namespace N2D2 {
/**
 * Class for methods for the FcCell type for the CPP_TensorRT export
 * FcCell, CPP_TensorRT EXPORT
*/
class CPP_TensorRT_FcCellExport : public FcCellExport,
                               public CPP_TensorRT_CellExport {
public:
    static void generate(FcCell& cell, const std::string& dirName);
    static void generateHeaderConstants(FcCell& cell, std::ofstream& header);
    static void generateHeaderFreeParameters(FcCell& cell,
                                             std::ofstream& header);

    static void generateHeaderBias(FcCell& cell, std::ofstream& header);
    static void generateHeaderBiasValues(FcCell& cell,
                                                std::ofstream& header);

    static void generateHeaderBiasVariable(FcCell& cell, std::ofstream& header);
    static void generateHeaderWeights(FcCell& cell, std::ofstream& header);
    static void generateHeaderWeightsSparse(FcCell& cell,
                                            std::ofstream& header);
    static void generateFileBias(FcCell& cell,
                                          std::ofstream& file);
    static void generateFileWeights(FcCell& cell, std::ofstream& file);

    static std::unique_ptr<CPP_TensorRT_FcCellExport> getInstance(Cell& cell);

    void generateCellProgramDescriptors(Cell& cell, std::ofstream& prog);
    void generateFcProgramTensorDesc(Cell& cell, std::ofstream& prog);
    void generateFcProgramLayerDesc(Cell& cell, std::ofstream& prog);
    void generateFcProgramFilterDesc(Cell& cell, std::ofstream& prog);
    void generateFcProgramActivationDesc(Cell& cell, std::ofstream& prog);

    void generateCellProgramInstanciateLayer(Cell& cell,
                                       std::vector<std::string>& parentsName,
                                       std::ofstream& prog);
    void generateCellProgramAllocateMemory(unsigned int targetIdx, std::ofstream& prog);

    void generateCellProgramInstanciateOutput(Cell& cell,
                                              unsigned int targetIdx,
                                               std::ofstream& prog);
    void generateFcProgramAddParam(Cell& cell,
                                   std::ofstream& prog);
    void generateFcProgramAddLayer(Cell& cell,
                                   std::vector<std::string>& parentsName,
                                   std::ofstream& prog);


private:
    static Registrar<FcCellExport> mRegistrar;
    static Registrar<CPP_TensorRT_CellExport> mRegistrarType;
};
}

#endif // N2D2_CPP_TensorRT_FCCELLEXPORT_H
