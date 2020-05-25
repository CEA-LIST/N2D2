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

#ifndef N2D2_CPP_TensorRT_DECONVCELLEXPORT_H
#define N2D2_CPP_TensorRT_DECONVCELLEXPORT_H

#include "Export/CPP_TensorRT/CPP_TensorRT_CellExport.hpp"
#include "Export/CPP_TensorRT/CPP_TensorRT_ConvCellExport.hpp"
#include "Export/CPP/CPP_DeconvCellExport.hpp"
#include "Export/DeconvCellExport.hpp"

namespace N2D2 {
/**
 * Class for methods for the DeconvCell type for the CPP_TensorRT export
 * DeconvCell, CPP_TensorRT EXPORT
*/
class CPP_TensorRT_DeconvCellExport : public DeconvCellExport,
                                 public CPP_TensorRT_CellExport {
public:
    static void generate(DeconvCell& cell, const std::string& dirName);
    static void generateHeaderConstants(DeconvCell& cell, std::ofstream& header);
    static void generateHeaderTensorRTConstants(DeconvCell& cell,std::ofstream& header);
    static void generateHeaderFreeParameters(DeconvCell& cell,
                                             std::ofstream& header);
    static void generateHeaderBias(DeconvCell& cell, std::ofstream& header);
    static void generateHeaderBiasVariable(DeconvCell& cell,
                                           std::ofstream& header);
    static void generateHeaderBiasValues(DeconvCell& cell,
                                          std::ofstream& header);
    static void generateHeaderWeights(DeconvCell& cell, std::ofstream& header);

    static void generateFileBias(DeconvCell& cell,
                                          std::ofstream& file);
    static void generateFileWeights(DeconvCell& cell, std::ofstream& file);

    static std::unique_ptr<CPP_TensorRT_DeconvCellExport> getInstance(Cell& cell);
    void generateCellProgramDescriptors(Cell& cell, std::ofstream& header);
    void generateDeconvProgramTensorDesc(Cell& cell, std::ofstream& header);
    void generateDeconvProgramLayerDesc(Cell& cell, std::ofstream& header);
    void generateDeconvProgramFilterDesc(Cell& cell, std::ofstream& header);
    void generateDeconvProgramActivationDesc(Cell& cell, std::ofstream& header);

    void generateCellProgramInstanciateLayer(Cell& cell,
                                       std::vector<std::string>& parentsName,
                                       std::ofstream& prog);
    void generateCellProgramAllocateMemory(unsigned int targetIdx, std::ofstream& prog);

    void generateCellProgramInstanciateOutput(Cell& cell,
                                              unsigned int targetIdx,
                                               std::ofstream& prog);
    void generateDeconvProgramAddLayer(Cell& cell,
                                     std::vector<std::string>& parentsName,
                                     std::ofstream& prog);

private:
    static Registrar<DeconvCellExport> mRegistrar;
    static Registrar<CPP_TensorRT_CellExport> mRegistrarType;
};
}

#endif // N2D2_CPP_TensorRT_DECONVCELLEXPORT_H
