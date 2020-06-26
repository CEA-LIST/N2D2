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

#ifndef N2D2_CPP_TensorRT_CONVCELLEXPORT_H
#define N2D2_CPP_TensorRT_CONVCELLEXPORT_H

#include "Export/CPP_TensorRT/CPP_TensorRT_CellExport.hpp"
#include "Export/CPP/CPP_ConvCellExport.hpp"
#include "Export/ConvCellExport.hpp"

namespace N2D2 {
/**
 * Class for methods for the ConvCell type for the CPP_TensorRT export
 * ConvCell, CPP_TensorRT EXPORT
*/
class CPP_TensorRT_ConvCellExport : public ConvCellExport,
                                 public CPP_TensorRT_CellExport {
public:
    static void generate(ConvCell& cell, const std::string& dirName);
    static void generateHeaderConstants(ConvCell& cell, std::ofstream& header);
    static void generateHeaderTensorRTConstants(ConvCell& cell,std::ofstream& header);
    static void generateHeaderFreeParameters(ConvCell& cell,
                                             std::ofstream& header);
    static void generateHeaderBias(ConvCell& cell, std::ofstream& header);
    static void generateHeaderBiasVariable(ConvCell& cell,
                                           std::ofstream& header);
    static void generateHeaderBiasValues(ConvCell& cell,
                                          std::ofstream& header);
    static void generateHeaderWeights(ConvCell& cell, std::ofstream& header);

    static void generateFileBias(ConvCell& cell,
                                          std::ofstream& file);
    static void generateFileWeights(ConvCell& cell, std::ofstream& file);

    static std::unique_ptr<CPP_TensorRT_ConvCellExport> getInstance(Cell& cell);

    void generateCellProgramDescriptors(Cell& cell, std::ofstream& header);
    void generateConvProgramTensorDesc(Cell& cell, std::ofstream& header);
    void generateConvProgramLayerDesc(Cell& cell, std::ofstream& header);
    void generateConvProgramFilterDesc(Cell& cell, std::ofstream& header);
    void generateConvProgramActivationDesc(Cell& cell, std::ofstream& header);

    void generateCellProgramInstanciateLayer(Cell& cell,
                                       std::vector<std::string>& parentsName,
                                       std::ofstream& prog);

    void generateCellProgramInstanciateOutput(Cell& cell,
                                              unsigned int targetIdx,
                                               std::ofstream& prog);
    void generateConvProgramAddParam(Cell& cell,
                                     std::ofstream& prog);
    void generateConvProgramAddLayer(Cell& cell,
                                     std::vector<std::string>& parentsName,
                                     std::ofstream& prog);

private:
    static Registrar<ConvCellExport> mRegistrar;
    static Registrar<CPP_TensorRT_CellExport> mRegistrarType;
};
}

#endif // N2D2_CPP_TensorRT_CONVCELLEXPORT_H
