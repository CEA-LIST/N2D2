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
#include "Cell/ScalingCell.hpp"
#include "Export/ScalingCellExport.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/CPP/CPP_ScalingCellExport.hpp"
#include "Export/C/C_CellExport.hpp"
#include "Export/C/C_ScalingCellExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

N2D2::Registrar<N2D2::ScalingCellExport>
N2D2::CPP_ScalingCellExport::mRegistrar(
    {"CPP", "CPP_ASMP", "CPP_STM32", "CPP_HLS"},
    N2D2::CPP_ScalingCellExport::generate);

N2D2::Registrar<N2D2::CPP_CellExport>
N2D2::CPP_ScalingCellExport::mRegistrarType(
    N2D2::ScalingCell::Type, N2D2::CPP_ScalingCellExport::getInstance);

void N2D2::CPP_ScalingCellExport::generate(const ScalingCell& cell, const std::string& dirName) {
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

void N2D2::CPP_ScalingCellExport::generateHeaderConstants(const ScalingCell& cell, std::ofstream& header) {
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n\n";

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*" 
                                                        << prefix << "_OUTPUTS_WIDTH*" 
                                                        << prefix << "_OUTPUTS_HEIGHT)\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix << "_NB_CHANNELS*" 
                                                         << prefix << "_CHANNELS_WIDTH*" 
                                                         << prefix << "_CHANNELS_HEIGHT)\n\n";

    CPP_CellExport::generateScaling(prefix, cell.getScaling(),
        DeepNetExport::isCellOutputUnsigned(cell), header);

    // TODO: needed for DNeuro_V2 emulator. Use the CPP way in the future?
    if (cell.getScaling().getMode() == ScalingMode::FLOAT_MULT
        || cell.getScaling().getMode() == ScalingMode::FIXED_MULT16
        || cell.getScaling().getMode() == ScalingMode::FIXED_MULT32)
    {
        C_ScalingCellExport::generateScaling(cell, header);
    }
}

std::unique_ptr<N2D2::CPP_ScalingCellExport>
N2D2::CPP_ScalingCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_ScalingCellExport>(new CPP_ScalingCellExport);
}

void N2D2::CPP_ScalingCellExport::generateCallCode(
    const DeepNet& deepNet,
    const Cell& cell, 
    std::stringstream& includes,
    std::stringstream& /*buffers*/, 
    std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    includes << "#include \"" << identifier << ".hpp\"\n";

    generateBenchmarkStart(deepNet, cell, functionCalls);

    const auto& parents = cell.getParentsCells();
    if(parents.empty()) {
        throw std::runtime_error("Cell must have a parent.");
    }

    const std::string inputBuffer
        = Utils::CIdentifier(parents[0] ? parents[0]->getName() + "_output"
                                        : "inputs");
    const std::string outputBuffer
        = Utils::CIdentifier(cell.getName() + "_output");

    functionCalls << "    scalingPropagate<" 
                << prefix << "_NB_CHANNELS, "
                << prefix << "_CHANNELS_HEIGHT, "
                << prefix << "_CHANNELS_WIDTH, "
                << prefix << "_NB_OUTPUTS, "
                << prefix << "_OUTPUTS_HEIGHT, " 
                << prefix << "_OUTPUTS_WIDTH, ";

    // Memory mapping: input
    const std::string parentIdentifier
        = Utils::CIdentifier((parents[0]) ? parents[0]->getName() : "env");
    const std::string parentPrefix
        = N2D2::Utils::upperCase(parentIdentifier);

    functionCalls << parentPrefix << "_MEM_CONT_OFFSET, "
        << parentPrefix << "_MEM_CONT_SIZE, "
        << parentPrefix << "_MEM_WRAP_OFFSET, "
        << parentPrefix << "_MEM_WRAP_SIZE, "
        << parentPrefix << "_MEM_STRIDE, ";

    // Memory mapping: output
    functionCalls << prefix << "_MEM_CONT_OFFSET, "
                << prefix << "_MEM_CONT_SIZE, "
                << prefix << "_MEM_WRAP_OFFSET, "
                << prefix << "_MEM_WRAP_SIZE, "
                << prefix << "_MEM_STRIDE"
            << ">("
                << inputBuffer << " , "
                << outputBuffer << " , "
                << prefix << "_SCALING"
            << ");\n\n";

    generateBenchmarkEnd(deepNet, cell, functionCalls);
    generateSaveOutputs(deepNet, cell, functionCalls);
}
