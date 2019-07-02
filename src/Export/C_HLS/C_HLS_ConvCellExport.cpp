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

#include "Export/C_HLS/C_HLS_ConvCellExport.hpp"

N2D2::Registrar<N2D2::ConvCellExport>
N2D2::C_HLS_ConvCellExport::mRegistrar("C_HLS",
                                       N2D2::C_HLS_ConvCellExport::generate);

N2D2::Registrar<N2D2::C_HLS_CellExport>
N2D2::C_HLS_ConvCellExport::mRegistrarType(
    ConvCell::Type, N2D2::C_HLS_ConvCellExport::getInstance);

void N2D2::C_HLS_ConvCellExport::generate(ConvCell& cell,
                                          const std::string& dirName)
{
    Utils::createDirectories(dirName + "/include");

    const std::string fileName = dirName + "/include/"
        + Utils::CIdentifier(cell.getName()) + ".h";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    C_CellExport::generateHeaderBegin(cell, header);
    C_CellExport::generateHeaderIncludes(cell, header);
    C_ConvCellExport::generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    C_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::C_HLS_ConvCellExport::generateHeaderFreeParameters(ConvCell& cell,
                                                              std::ofstream
                                                              & header)
{
    C_ConvCellExport::generateHeaderBias(cell, header);
    generateHeaderWeights(cell, header);
}

void N2D2::C_HLS_ConvCellExport::generateHeaderWeights(ConvCell& cell,
                                                       std::ofstream& header)
{
    // NOTE: GCC 4.4 doesn't completly support C++11 initializer list
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    // Full weights storage format
    header << "typedef WDATA_T " << prefix << "_KERNEL_FULL_T"
                                              "[" << prefix
           << "_KERNEL_HEIGHT][" << prefix << "_KERNEL_WIDTH];\n";
    header << "static const " << prefix << "_KERNEL_FULL_T " << identifier
           << "_weights"
              "[" << prefix << "_NB_OUTPUTS][" << prefix
           << "_NB_CHANNELS] = {\n";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ",\n";

        header << "    {";

        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (channel > 0)
                header << ", ";

            if (!cell.isConnection(channel, output))
                header << "{{0}}";
            else {
                Tensor<Float_T> kernel;
                cell.getWeight(output, channel, kernel);

                header << "{";

                for (unsigned int sy = 0; sy < cell.getKernelHeight(); ++sy) {
                    if (sy > 0)
                        header << ",\n";

                    header << "{";

                    for (unsigned int sx = 0; sx < cell.getKernelWidth();
                         ++sx) {
                        if (sx > 0)
                            header << ", ";

                        CellExport::generateFreeParameter(
                            cell,
                            kernel(sx, sy),
                            header);
                    }

                    header << "}";
                }

                header << "}";
            }
        }

        header << "}";
    }

    header << "};\n\n";

    // Compact weights storage format
    unsigned int nbKernels = 0;

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (cell.isConnection(channel, output))
                ++nbKernels;
        }
    }

    header << "#define " << prefix << "_NB_KERNELS " << nbKernels << "\n\n";

    header << "static const WDATA_T " << identifier << "_weights_compact"
                                                           "[" << prefix
           << "_NB_KERNELS][" << prefix << "_KERNEL_HEIGHT][" << prefix
           << "_KERNEL_WIDTH] = {\n";

    // Weights
    nbKernels = 0;

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (!cell.isConnection(channel, output))
                continue;

            Tensor<Float_T> kernel;
            cell.getWeight(output, channel, kernel);

            if (nbKernels > 0)
                header << ",\n";

            header << "{\n";

            for (unsigned int sy = 0; sy < cell.getKernelHeight(); ++sy) {
                if (sy > 0)
                    header << ",\n";

                header << "    {";

                for (unsigned int sx = 0; sx < cell.getKernelWidth(); ++sx) {
                    if (sx > 0)
                        header << ", ";

                    CellExport::generateFreeParameter(
                        cell, kernel(sx, sy), header);
                }

                header << "}";
            }

            header << "}";

            ++nbKernels;
        }
    }

    header << "};\n";

    header << "static const int " << identifier << "_weights_compact_map"
                                                       "[" << prefix
           << "_NB_OUTPUTS][" << prefix << "_NB_CHANNELS] = {\n";

    nbKernels = 0;

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ",\n";

        header << "    {";

        for (unsigned int channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (channel > 0)
                header << ", ";

            if (!cell.isConnection(channel, output))
                header << "-1";
            else {
                header << nbKernels;
                ++nbKernels;
            }
        }

        header << "}";
    }

    header << "};\n\n";
}

std::unique_ptr<N2D2::C_HLS_ConvCellExport>
N2D2::C_HLS_ConvCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<C_HLS_ConvCellExport>(new C_HLS_ConvCellExport);
}

N2D2::C_HLS_CellExport::TclDirectives
N2D2::C_HLS_ConvCellExport::getTclDirectives(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& /*parentCells*/,
    bool isUnsigned)
{
    std::stringstream funcName;
    funcName << Utils::CIdentifier(cell.getName()) << "_"
             << ((isUnsigned) ? "u" : "") << "propagate";

    const ConvCell* convCell = dynamic_cast<ConvCell*>(&cell);

    return TclDirectives(funcName.str(),
                         "Conv",
                         cell.getNbChannels(),
                         cell.getChannelsWidth(),
                         cell.getChannelsHeight(),
                         convCell->getKernelWidth(),
                         convCell->getKernelHeight());
}

void N2D2::C_HLS_ConvCellExport::generateCellPrototype(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& /*parentCells*/,
    const std::string& outputSizeName,
    std::ofstream& prog,
    bool isUnsigned)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "CONVCELL_" << ((isUnsigned) ? "U" : "") << "PROPAGATE"
        << ((!cell.isFullMap()) ? "_COMPACT" : "") << "("
        << identifier << ", "
        << prefix << "_NB_CHANNELS, "
        << prefix << "_CHANNELS_HEIGHT, "
        << prefix << "_CHANNELS_WIDTH, "
        << outputSizeName << ", "
        << prefix << "_OUTPUTS_HEIGHT, "
        << prefix << "_OUTPUTS_WIDTH, "
        << prefix << "_NB_KERNELS, "
        << prefix << "_KERNEL_HEIGHT, "
        << prefix << "_KERNEL_WIDTH)\n";
}
