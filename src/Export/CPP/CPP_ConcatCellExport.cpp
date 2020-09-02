/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <memory>
#include <ostream>
#include <string>
#include <stdexcept>

#include "DeepNet.hpp"
#include "Cell/Cell.hpp"
#include "Export/CellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/MemoryManager.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/CPP/Cells/CPP_ConcatCell.hpp"
#include "Export/CPP/CPP_ConcatCellExport.hpp"
#include "Export/CPP/CPP_Config.hpp"
#include "utils/Utils.hpp"

// Specific cell to the CPP export, no Registrar to a generate

N2D2::Registrar<N2D2::CPP_CellExport> N2D2::CPP_ConcatCellExport::mRegistrarType(
        N2D2::CPP_ConcatCell::Type, N2D2::CPP_ConcatCellExport::getInstance);

void N2D2::CPP_ConcatCellExport::generate(const CPP_ConcatCell& cell,
                                        const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_ConcatCellExport::generateHeaderConstants(const CPP_ConcatCell& cell,
                                                       std::ofstream& header)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    const auto parentsCells = cell.getParentsCells();
    header << "#define " << prefix << "_NB_INPUTS " << parentsCells.size() << "\n";

    header << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n"
           << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"

           << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*" 
                                                        << prefix << "_OUTPUTS_WIDTH*" 
                                                        << prefix << "_OUTPUTS_HEIGHT)" << "\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix  << "_NB_CHANNELS*" 
                                                         << prefix << "_CHANNELS_WIDTH*"
                                                         << prefix << "_CHANNELS_HEIGHT)" << "\n"
           << "\n";
}

void N2D2::CPP_ConcatCellExport::generateCallCode(
    const DeepNet& deepNet,
    const Cell& cell, 
    std::stringstream& includes,
    std::stringstream& /*buffers*/, 
    std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    // includes
    includes << "#include \"" << identifier << ".hpp\"\n";

    // functionCalls: input/output buffers
    const std::string inputType = DeepNetExport::isCellInputsUnsigned(cell)
        ? "UDATA_T" : "DATA_T";

    const auto& parents = deepNet.getParentCells(cell.getName());

    std::ostringstream inputBufferStr;

    for (const auto& parentCell: parents) {
        inputBufferStr << ", " << Utils::CIdentifier(parentCell->getName()
                                                        + "_output");
    }

    const std::string inputBuffer = inputBufferStr.str();
    const std::string outputBuffer = Utils::CIdentifier(cell.getName() + "_output");

    generateBenchmarkStart(deepNet, cell, functionCalls);

    // functionCalls: propagate
    // concatenatePropagate is a variadic template
    functionCalls << "    concatenatePropagate<" 
                        << prefix << "_NB_INPUTS, "
                        << prefix << "_CHANNELS_HEIGHT, "
                        << prefix << "_CHANNELS_WIDTH, "
                        << prefix << "_NB_OUTPUTS, "
                        << prefix << "_OUTPUTS_HEIGHT, " 
                        << prefix << "_OUTPUTS_WIDTH, "
                        // Memory mapping: output
                        << prefix << "_MEM_CONT_OFFSET, "
                        << prefix << "_MEM_CONT_SIZE, "
                        << prefix << "_MEM_WRAP_OFFSET, " 
                        << prefix << "_MEM_WRAP_SIZE, " 
                        << prefix << "_MEM_STRIDE";

    for (const auto& parentCell: parents) {
        const std::string parentIdentifier
            = Utils::CIdentifier((parentCell) ? parentCell->getName() : "env");
        const std::string parentPrefix
            = N2D2::Utils::upperCase(parentIdentifier);

        // Memory mapping: inputs
        functionCalls << ", "
            << parentPrefix << "_NB_OUTPUTS, "
            << parentPrefix << "_MEM_CONT_OFFSET, "
            << parentPrefix << "_MEM_CONT_SIZE, "
            << parentPrefix << "_MEM_WRAP_OFFSET, "
            << parentPrefix << "_MEM_WRAP_SIZE, "
            << parentPrefix << "_MEM_STRIDE";
    }

    functionCalls << ">("
                        << outputBuffer
                        << inputBuffer
                    << ");\n\n";

    generateBenchmarkEnd(deepNet, cell, functionCalls);
    generateSaveOutputs(deepNet, cell, functionCalls);
}

std::unique_ptr<N2D2::CPP_ConcatCellExport>
N2D2::CPP_ConcatCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_ConcatCellExport>(new CPP_ConcatCellExport);
}
