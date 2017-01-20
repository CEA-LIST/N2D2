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

#include "Export/CPP/CPP_ConvCellExport.hpp"

N2D2::Registrar<N2D2::ConvCellExport>
N2D2::CPP_ConvCellExport::mRegistrar("CPP", N2D2::CPP_ConvCellExport::generate);

void N2D2::CPP_ConvCellExport::generate(ConvCell& cell,
                                        const std::string& dirName)
{
    Utils::createDirectories(dirName + "/include");

    const std::string fileName = dirName + "/include/" + cell.getName()
                                 + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    C_CellExport::generateHeaderBegin(cell, header);
    C_ConvCellExport::generateHeaderIncludes(cell, header);
    C_ConvCellExport::generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    C_ConvCellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderFreeParameters(ConvCell& cell,
                                                            std::ofstream
                                                            & header)
{
    generateHeaderBias(cell, header);
    generateHeaderWeights(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderBias(ConvCell& cell,
                                                  std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    C_ConvCellExport::generateHeaderBiasValues(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderBiasVariable(ConvCell& cell,
                                                          std::ofstream& header)
{
    header << "const std::vector<WDATA_T> " << cell.getName() << "_biases = ";
}

void N2D2::CPP_ConvCellExport::generateHeaderWeights(ConvCell& cell,
                                                     std::ofstream& header)
{
    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (!cell.isConnection(channel, output))
                continue;

            generateHeaderKernelWeightsVariable(cell, header, output, channel);
            C_ConvCellExport::generateHeaderKernelWeightsValues(
                cell, header, output, channel);
        }
    }

    generateHeaderWeightsVariable(cell, header);
    C_ConvCellExport::generateHeaderWeightsValues(cell, header);
}

void N2D2::CPP_ConvCellExport::generateHeaderKernelWeightsVariable(
    ConvCell& cell,
    std::ofstream& header,
    unsigned int output,
    unsigned int channel)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    header << "const std::vector<std::vector<WDATA_T> > " << cell.getName()
           << "_weights_" << output << "_" << channel << " = ";
}

void N2D2::CPP_ConvCellExport::generateHeaderWeightsVariable(ConvCell& cell,
                                                             std::ofstream
                                                             & header)
{
    const std::string prefix = Utils::upperCase(cell.getName());

    header << "typedef const std::vector<std::vector<WDATA_T> > " << prefix
           << "_KERNEL_T;\n";
    header << "const std::vector<std::vector<" << prefix << "_KERNEL_T*> > "
           << cell.getName() << "_weights = ";
}
