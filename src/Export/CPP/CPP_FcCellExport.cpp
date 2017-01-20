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

#include "Export/CPP/CPP_FcCellExport.hpp"

N2D2::Registrar<N2D2::FcCellExport>
N2D2::CPP_FcCellExport::mRegistrar("CPP", N2D2::CPP_FcCellExport::generate);

void N2D2::CPP_FcCellExport::generate(FcCell& cell, const std::string& dirName)
{
    Utils::createDirectories(dirName + "/include");

    const std::string fileName = dirName + "/include/" + cell.getName()
                                 + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    C_CellExport::generateHeaderBegin(cell, header);
    C_FcCellExport::generateHeaderIncludes(cell, header);
    C_FcCellExport::generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    C_FcCellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderFreeParameters(FcCell& cell,
                                                          std::ofstream& header)
{
    generateHeaderBias(cell, header);

    if (mThreshold > 0.0)
        C_FcCellExport::generateHeaderWeightsSparse(cell, header);
    else
        generateHeaderWeights(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderBias(FcCell& cell,
                                                std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    C_FcCellExport::generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderBiasVariable(FcCell& cell,
                                                        std::ofstream& header)
{
    header << "const std::vector<WDATA_T> " << cell.getName() << "_biases = ";
}

void N2D2::CPP_FcCellExport::generateHeaderWeights(FcCell& cell,
                                                   std::ofstream& header)
{
    generateHeaderWeightsVariable(cell, header);
    C_FcCellExport::generateHeaderWeightsValues(cell, header);
}

void N2D2::CPP_FcCellExport::generateHeaderWeightsVariable(FcCell& cell,
                                                           std::ofstream
                                                           & header)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    header << "#define " << prefix << "_NB_WEIGHTS (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n\n";

    // Weights
    header << "const std::vector<std::vector<WDATA_T> > " << cell.getName()
           << "_weights = \n";
}
