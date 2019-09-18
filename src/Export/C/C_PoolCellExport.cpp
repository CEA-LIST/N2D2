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

#include "Export/C/C_PoolCellExport.hpp"

N2D2::Registrar<N2D2::PoolCellExport>
N2D2::C_PoolCellExport::mRegistrar("C", N2D2::C_PoolCellExport::generate);

N2D2::Registrar<N2D2::C_CellExport>
N2D2::C_PoolCellExport::mRegistrarType(PoolCell::Type,
                                       N2D2::C_PoolCellExport::getInstance);

void N2D2::C_PoolCellExport::generate(const PoolCell& cell, const std::string& dirName) {
    Utils::createDirectories(dirName + "/include");

    const std::string fileName = dirName + "/include/"
        + Utils::CIdentifier(cell.getName()) + ".h";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    C_CellExport::generateHeaderBegin(cell, header);
    C_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);

    if (!cell.isUnitMap())
        generateHeaderConnections(cell, header);

    C_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::C_PoolCellExport::generateHeaderConstants(const PoolCell& cell, std::ofstream& header) {
    // Constants
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT "  << cell.getChannelsHeight() << "\n"
           << "#define " << prefix << "_POOL_WIDTH " << cell.getPoolWidth() << "\n"
           << "#define " << prefix << "_POOL_HEIGHT " << cell.getPoolHeight() << "\n"
           << "#define " << prefix << "_PADDING_X " << cell.getPaddingX() << "\n"
           << "#define " << prefix << "_PADDING_Y " << cell.getPaddingY() << "\n"
           << "#define " << prefix << "_STRIDE_X " << cell.getStrideX() << "\n"
           << "#define " << prefix << "_STRIDE_Y " << cell.getStrideY() << "\n"
           << "#define " << prefix << "_POOLING " << cell.getPooling() << "\n\n";

    C_CellExport::generateActivation(cell, header);
    C_CellExport::generateActivationScaling(cell, header);
}

void N2D2::C_PoolCellExport::generateHeaderConnections(const PoolCell& cell, std::ofstream& header) {
    generateHeaderConnectionsVariable(cell, header);
    generateHeaderConnectionsValues(cell, header);
}

void N2D2::C_PoolCellExport::generateHeaderConnectionsVariable(const PoolCell& cell, std::ofstream& header) {
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "static const char " << identifier << "_mapping[" << prefix
           << "_NB_OUTPUTS][" << prefix << "_NB_CHANNELS] = ";
}

void N2D2::C_PoolCellExport::generateHeaderConnectionsValues(const PoolCell& cell, std::ofstream& header) {
    header << "{";

    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ",\n";

        header << "    {";

        for (std::size_t channel = 0; channel < cell.getNbChannels(); ++channel) {
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

std::unique_ptr<N2D2::C_PoolCellExport>
N2D2::C_PoolCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<C_PoolCellExport>(new C_PoolCellExport);
}

void N2D2::C_PoolCellExport::generateCellData(Cell& cell,
                                              const std::string& outputName,
                                              const std::string& outputSizeName,
                                              std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "static DATA_T " << outputName << "[" << outputSizeName << "]["
         << prefix << "_OUTPUTS_HEIGHT][" << prefix << "_OUTPUTS_WIDTH];\n";
}

void N2D2::C_PoolCellExport::generateCellFunction(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& /*parentCells*/,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& outputSizeName,
    std::ofstream& prog,
    bool isUnsigned,
    const std::string& funcProto,
    const std::string& memProto,
    bool /*memCompact*/)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto = (funcProto.empty()) ? "poolcell" : funcProto;

    // Time analysis (start)
    prog << "#ifdef TIME_ANALYSIS\n"
        "   gettimeofday(&start, NULL);\n"
        "#endif\n";

    prog << "    " << proto << "_" << ((isUnsigned) ? "u" : "") << "propagate"
        << ((cell.isUnitMap()) ? "_unitmap" : "") << "("
        << prefix << "_NB_CHANNELS, "
        << prefix << "_CHANNELS_HEIGHT, "
        << prefix << "_CHANNELS_WIDTH, "
        << prefix << "_STRIDE_Y, "
        << prefix << "_STRIDE_X, "
        << inputName << ", "
        << outputSizeName << ", "
        << prefix << "_OUTPUTS_HEIGHT, "
        << prefix << "_OUTPUTS_WIDTH, "
        << prefix << "_NB_OUTPUTS, "
        << prefix << "_OUTPUT_OFFSET, "
        << outputName << ", "
        << prefix << "_POOL_HEIGHT, "
        << prefix << "_POOL_WIDTH, ";

    if (!cell.isUnitMap())
        prog << memProto << identifier << "_mapping, ";

    prog << prefix << "_POOLING, "
        << prefix << "_ACTIVATION, "
        << prefix << "_SHIFT);\n";

    // Time analysis (end)
    prog << "#ifdef TIME_ANALYSIS\n"
        "    gettimeofday(&end, NULL);\n"
        "    static RUNNING_MEAN_T " << identifier << "_timing = {0.0, 0};\n"
        "    time_analysis(\"" << identifier << "\", start, end, &"
        << identifier << "_timing);\n"
        "#endif\n";
}

void N2D2::C_PoolCellExport::generateOutputFunction(Cell& cell,
                                                    const std::string
                                                    & inputName,
                                                    const std::string
                                                    & outputName,
                                                    std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    if ((cell.getOutputsWidth() == 1) && (cell.getOutputsHeight() == 1)) {
        prog << "\n"
                "    output_max(" << prefix << "_NB_OUTPUTS, " << inputName
             << ", " << outputName << ");\n";
    } else {
        prog << "\n"
                "    spatial_output_max(" << prefix << "_NB_OUTPUTS, " << prefix
             << "_OUTPUTS_HEIGHT, " << prefix << "_OUTPUTS_WIDTH, " << inputName
             << ", " << outputName << ");\n";
    }
}
