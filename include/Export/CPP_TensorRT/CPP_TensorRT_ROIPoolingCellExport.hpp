/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_CPP_TensorRT_ROIPOOLINGCELLEXPORT_H
#define N2D2_CPP_TensorRT_ROIPOOLINGCELLEXPORT_H

#include "Export/ROIPoolingCellExport.hpp"
#include "Export/CPP/CPP_ROIPoolingCellExport.hpp"
#include "Export/CPP_TensorRT/CPP_TensorRT_CellExport.hpp"

namespace N2D2 {
/**
 * Class for methods for the ROI Pooling type for the CPP_TensorRT export
 * ROIPoolingCell, CPP_TensorRT EXPORT
*/
class CPP_TensorRT_ROIPoolingCellExport : public ROIPoolingCellExport,
                                 public CPP_TensorRT_CellExport {
public:
    static void generate(ROIPoolingCell& cell, const std::string& dirName);

    static std::unique_ptr<CPP_TensorRT_ROIPoolingCellExport> getInstance(Cell& cell);

    // Commun methods for all cells

    void generateCellProgramDescriptors(Cell& cell, std::ofstream& prog);

    void generateCellProgramInstanciateLayer(Cell& cell,
                                            std::vector<std::string>& parentsName,
                                            std::ofstream& prog);
    void generateCellProgramAllocateMemory(unsigned int targetIdx, std::ofstream& prog);

    void generateCellProgramInstanciateOutput(Cell& cell,
                                               unsigned int targetIdx,
                                               std::ofstream& prog);

    void generateROIPoolingProgramAddLayer(Cell& cell,
                                     std::vector<std::string>& parentsName,
                                     std::ofstream& prog);


private:
    static Registrar<ROIPoolingCellExport> mRegistrar;
    static Registrar<CPP_TensorRT_CellExport> mRegistrarType;
};
}

#endif // N2D2_CPP_TensorRT_ROIPOOLINGCELLEXPORT_H



