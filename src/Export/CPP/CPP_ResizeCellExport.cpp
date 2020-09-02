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

#include "DeepNet.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/ResizeCell.hpp"
#include "Export/ResizeCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/CPP/CPP_ConvCellExport.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/CPP/CPP_ResizeCellExport.hpp"
#include "Export/CPP/CPP_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

N2D2::Registrar<N2D2::ResizeCellExport> N2D2::CPP_ResizeCellExport::mRegistrar(
    {"CPP", "CPP_STM32"},
    N2D2::CPP_ResizeCellExport::generate);

N2D2::Registrar<N2D2::CPP_CellExport> N2D2::CPP_ResizeCellExport::mRegistrarType(
        N2D2::ResizeCell::Type, N2D2::CPP_ResizeCellExport::getInstance);

void N2D2::CPP_ResizeCellExport::generate(ResizeCell& cell,
                                             const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_ResizeCellExport::generateHeaderConstants(ResizeCell& cell,
                                                            std::ofstream
                                                            & header)
{
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

    header << "#define " << prefix << "_RESIZE_MODE " << cell.getMode() << "\n"
           << "#define " << prefix << "_ALIGN_CORNER " << cell.isAlignedCorner() << "\n"
           << std::endl;
}

std::unique_ptr<N2D2::CPP_ResizeCellExport>
N2D2::CPP_ResizeCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_ResizeCellExport>(new CPP_ResizeCellExport);
}

void N2D2::CPP_ResizeCellExport::generateCallCode(
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

    const auto& parents = deepNet.getParentCells(cell.getName());
    const std::string inputBuffer
        = Utils::CIdentifier(parents[0] ? parents[0]->getName() + "_output"
                                        : "inputs");
    const std::string outputBuffer
        = Utils::CIdentifier(cell.getName() + "_output");

    functionCalls << "    resizeNearestNeighborPropagate<" 
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
                << outputBuffer
            << ");\n\n";

    generateBenchmarkEnd(deepNet, cell, functionCalls);
    generateSaveOutputs(deepNet, cell, functionCalls);
}
