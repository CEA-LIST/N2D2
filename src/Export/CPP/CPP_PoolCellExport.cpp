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

#include "DeepNet.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/PoolCell.hpp"
#include "Export/PoolCellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/CPP/CPP_ConvCellExport.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/CPP/CPP_PoolCellExport.hpp"
#include "Export/CPP/CPP_DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

N2D2::Registrar<N2D2::PoolCellExport> N2D2::CPP_PoolCellExport::mRegistrar(
    {"CPP", "CPP_ASMP", "CPP_STM32", "CPP_HLS"},
    N2D2::CPP_PoolCellExport::generate);

N2D2::Registrar<N2D2::CPP_CellExport> N2D2::CPP_PoolCellExport::mRegistrarType(
    N2D2::PoolCell::Type, N2D2::CPP_PoolCellExport::getInstance);

void N2D2::CPP_PoolCellExport::generate(const PoolCell& cell,
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
    //generateHeaderConnections(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_PoolCellExport::generateHeaderConstants(const PoolCell& cell,
                                                       std::ofstream& header)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    // Handle extended padding
    std::vector<int> padding = cell.getExtendedPadding();
    padding[0] += cell.getPaddingX();  // X_L
    padding[1] += cell.getPaddingY();  // Y_T
    padding[2] += cell.getPaddingX();  // X_R
    padding[3] += cell.getPaddingY();  // Y_B

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

    header << "#define " << prefix << "_POOL_WIDTH " << cell.getPoolWidth() << "\n"
           << "#define " << prefix << "_POOL_HEIGHT " << cell.getPoolHeight() << "\n"
           << "#define " << prefix << "_PADDING_X " << padding[0] << "\n"
           << "#define " << prefix << "_PADDING_Y " << padding[1] << "\n"
           << "#define " << prefix << "_STRIDE_X " << cell.getStrideX() << "\n"
           << "#define " << prefix << "_STRIDE_Y " << cell.getStrideY() << "\n"
           << "#define " << prefix << "_POOLING " << cell.getPooling() << "\n\n";

    CPP_CellExport::generateActivation(cell, header);
    CPP_CellExport::generateActivationScaling(cell, header);
}

void N2D2::CPP_PoolCellExport::generateHeaderConnections(const PoolCell& cell,
                                                         std::ofstream& header)
{
    header << "const std::vector<std::vector<char> > "
        << Utils::CIdentifier(cell.getName()) << "_mapping = {";

    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ",\n";

        header << "    {";

        for (std::size_t channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (channel > 0)
                header << ", ";

            if (!cell.isConnection(channel, output))
                header << "0";
            else
                header << "1";
        }

        header << "}";
    }

    header << "};\n\n";
}

std::unique_ptr<N2D2::CPP_PoolCellExport>
N2D2::CPP_PoolCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_PoolCellExport>(new CPP_PoolCellExport);
}

void N2D2::CPP_PoolCellExport::generateCallCode(
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

    functionCalls << "    poolcellPropagate<"
                << prefix << "_NB_CHANNELS, "
                << prefix << "_CHANNELS_HEIGHT, "
                << prefix << "_CHANNELS_WIDTH, "
                << prefix << "_NB_OUTPUTS, "
                << prefix << "_OUTPUTS_HEIGHT, " 
                << prefix << "_OUTPUTS_WIDTH, "
                << prefix << "_PADDING_Y, "
                << prefix << "_PADDING_X, "
                << prefix << "_STRIDE_Y, "
                << prefix << "_STRIDE_X, "
                << prefix << "_POOL_HEIGHT, "
                << prefix << "_POOL_WIDTH, "
                << prefix << "_POOLING, "
                << prefix << "_ACTIVATION, ";

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
