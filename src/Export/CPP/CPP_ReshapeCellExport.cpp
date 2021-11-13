/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include <string>
#include "Cell/ReshapeCell.hpp"
#include "Export/ReshapeCellExport.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/CPP/CPP_ReshapeCellExport.hpp"
#include "Export/C/C_CellExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

N2D2::Registrar<N2D2::ReshapeCellExport>
N2D2::CPP_ReshapeCellExport::mRegistrar(
    {"CPP", "CPP_ASMP", "CPP_STM32", "CPP_HLS"},
    N2D2::CPP_ReshapeCellExport::generate);

N2D2::Registrar<N2D2::CPP_CellExport>
N2D2::CPP_ReshapeCellExport::mRegistrarType(
    N2D2::ReshapeCell::Type, N2D2::CPP_ReshapeCellExport::getInstance);

void N2D2::CPP_ReshapeCellExport::generate(const ReshapeCell& cell, const std::string& dirName) {
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/" + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());
    if(!header.good()) {
        throw std::runtime_error("Could not create CPP header file: " + fileName);
    }

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);

    if(!header) {
        throw std::runtime_error("Error writing file: " + fileName);
    }
}

void N2D2::CPP_ReshapeCellExport::generateHeaderConstants(const ReshapeCell& cell, std::ofstream& header) {
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    const std::vector<size_t>& shape = cell.getOutputsDims();

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n"
           << "const int " << prefix << "_SHAPE[" << shape.size() << "] = {"
            << Utils::join(shape.begin(), shape.end(), ',') << "};\n\n";

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*" 
                                                        << prefix << "_OUTPUTS_WIDTH*" 
                                                        << prefix << "_OUTPUTS_HEIGHT)\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix << "_NB_CHANNELS*" 
                                                         << prefix << "_CHANNELS_WIDTH*" 
                                                         << prefix << "_CHANNELS_HEIGHT)\n\n";
}

std::unique_ptr<N2D2::CPP_ReshapeCellExport>
N2D2::CPP_ReshapeCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_ReshapeCellExport>(new CPP_ReshapeCellExport);
}

void N2D2::CPP_ReshapeCellExport::generateCallCode(
    const DeepNet& /*deepNet*/,
    const Cell& cell, 
    std::stringstream& includes,
    std::stringstream& /*buffers*/, 
    std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    includes << "#include \"" << identifier << ".hpp\"\n";

    functionCalls << "    // reshape: " << prefix << " -- nothing to do!\n";
}
