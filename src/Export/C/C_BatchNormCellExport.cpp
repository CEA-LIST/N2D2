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

#include "Export/C/C_BatchNormCellExport.hpp"

N2D2::Registrar<N2D2::BatchNormCellExport>
N2D2::C_BatchNormCellExport::mRegistrar("C",
                                        N2D2::C_BatchNormCellExport::generate);

N2D2::Registrar<N2D2::C_CellExport> N2D2::C_BatchNormCellExport::mRegistrarType(
    BatchNormCell::Type, N2D2::C_BatchNormCellExport::getInstance);

void N2D2::C_BatchNormCellExport::generate(BatchNormCell& cell,
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
    generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    C_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::C_BatchNormCellExport::generateHeaderConstants(BatchNormCell& cell,
                                                          std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs()
           << "\n"
              "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels()
           << "\n"
              "#define " << prefix << "_OUTPUTS_WIDTH "
           << cell.getOutputsWidth() << "\n"
                                        "#define " << prefix
           << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
                                                               "#define "
           << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth()
           << "\n"
              "#define " << prefix << "_CHANNELS_HEIGHT "
           << cell.getChannelsHeight() << "\n\n";

    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);

    if (cellFrame != NULL) {
        header << "#define " << prefix << "_ACTIVATION "
               << ((cellFrame->getActivation())
                       ? cellFrame->getActivation()->getType()
                       : "Linear") << "\n";
    }
}

void N2D2::C_BatchNormCellExport::generateHeaderFreeParameters(BatchNormCell
                                                                 & cell,
                                                                 std::ofstream
                                                                 & header)
{
    generateHeaderEpsilon(cell, header);
    generateHeaderBias(cell, header);
    generateHeaderVariance(cell, header);
    generateHeaderMean(cell, header);
    generateHeaderScale(cell, header);
}

void N2D2::C_BatchNormCellExport::generateHeaderEpsilon(BatchNormCell& cell,
                                                          std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    header << "static double " << prefix
           << "_EPSILON = " << cell.getParameter<double>("Epsilon") << ";\n";
}

void N2D2::C_BatchNormCellExport::generateHeaderBias(BatchNormCell& cell,
                                                       std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    header << "static WDATA_T " << identifier << "_biases["
           << Utils::upperCase(identifier) << "_NB_OUTPUTS] = ";

    header << "{";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        Tensor<double> bias;
        cell.getBias(output, bias);

        CellExport::generateFreeParameter(
            cell, bias(0), header);
    }
    header << "};\n";
}

void N2D2::C_BatchNormCellExport::generateHeaderVariance(BatchNormCell& cell,
                                                           std::ofstream
                                                           & header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "static WDATA_T " << identifier << "_variances[" << prefix
           << "_NB_OUTPUTS] = {\n";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        Tensor<double> variance;
        cell.getVariance(output, variance);

        CellExport::generateFreeParameter(
            cell, variance(0), header);
    }
    header << "};\n\n";
}

void N2D2::C_BatchNormCellExport::generateHeaderMean(BatchNormCell& cell,
                                                       std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "static WDATA_T " << identifier << "_means[" << prefix
           << "_NB_OUTPUTS] = {\n";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        Tensor<double> mean;
        cell.getMean(output, mean);

        CellExport::generateFreeParameter(
            cell, mean(0), header);
    }
    header << "};\n\n";
}

void N2D2::C_BatchNormCellExport::generateHeaderScale(BatchNormCell& cell,
                                                        std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "static WDATA_T " << identifier << "_scales[" << prefix
           << "_NB_OUTPUTS] = {\n";

    for (unsigned int output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        Tensor<double> scale;
        cell.getScale(output, scale);

        CellExport::generateFreeParameter(
            cell, scale(0), header);
    }

    header << "};\n\n";
}

std::unique_ptr<N2D2::C_BatchNormCellExport>
N2D2::C_BatchNormCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<C_BatchNormCellExport>(new C_BatchNormCellExport);
}

void N2D2::C_BatchNormCellExport::generateCellData(Cell& cell,
                                                   const std::string
                                                   & outputName,
                                                   const std::string
                                                   & outputSizeName,
                                                   std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "static DATA_T " << outputName << "[" << outputSizeName << "]["
         << prefix << "_OUTPUTS_HEIGHT][" << prefix << "_OUTPUTS_WIDTH];\n";
}

void N2D2::C_BatchNormCellExport::generateCellFunction(
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
    const std::string proto = (funcProto.empty()) ? "batchnormcell" : funcProto;

    prog << "    " << proto << "_" << ((isUnsigned) ? "u" : "") << "propagate("
        << prefix << "_NB_CHANNELS, "
        << prefix << "_CHANNELS_HEIGHT, "
        << prefix << "_CHANNELS_WIDTH, "
        << inputName << ", "
        << outputSizeName << ", "
        << prefix << "_OUTPUT_OFFSET, "
        << outputName << ", "
        << memProto << identifier << "_biases, "
        << memProto << identifier << "_variances, "
        << memProto << identifier << "_means, "
        << memProto << identifier << "_scales, "
        << prefix << "_EPSILON, "
        << prefix << "_ACTIVATION);\n ";
}

void N2D2::C_BatchNormCellExport::generateOutputFunction(Cell& cell,
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
