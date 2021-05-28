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

#include "Export/C/C_ConvCellExport.hpp"

N2D2::Registrar<N2D2::ConvCellExport>
N2D2::C_ConvCellExport::mRegistrar("C", N2D2::C_ConvCellExport::generate);

N2D2::Registrar<N2D2::C_CellExport>
N2D2::C_ConvCellExport::mRegistrarType(ConvCell::Type,
                                       N2D2::C_ConvCellExport::getInstance);

void N2D2::C_ConvCellExport::generate(const ConvCell& cell,
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

void N2D2::C_ConvCellExport::generateHeaderConstants(const ConvCell& cell, std::ofstream& header) {
    // Constants
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    // Handle extended padding
    std::vector<int> padding = cell.getExtendedPadding();
    padding[0] += cell.getPaddingX();  // X_L
    padding[1] += cell.getPaddingY();  // Y_T
    padding[2] += cell.getPaddingX();  // X_R
    padding[3] += cell.getPaddingY();  // Y_B

    const std::size_t oxSize = (std::size_t) (
        (cell.getChannelsWidth() + padding[0] + padding[2] - cell.getKernelWidth() + cell.getStrideX())/
        static_cast<double>(cell.getStrideX())
    );

    const std::size_t oySize = (std::size_t)(
        (cell.getChannelsHeight() + padding[1] + padding[3] - cell.getKernelHeight() + cell.getStrideY())/
        static_cast<double>(cell.getStrideY())
    );

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_OX_SIZE " << oxSize << "\n"
           << "#define " << prefix << "_OY_SIZE " << oySize << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n"
           << "#define " << prefix << "_KERNEL_WIDTH " << cell.getKernelWidth() << "\n"
           << "#define " << prefix << "_KERNEL_HEIGHT " << cell.getKernelHeight() << "\n"
           << "#define " << prefix << "_SUB_SAMPLE_X " << cell.getSubSampleX() << "\n"
           << "#define " << prefix << "_SUB_SAMPLE_Y " << cell.getSubSampleY() << "\n"
           << "#define " << prefix << "_STRIDE_X " << cell.getStrideX() << "\n"
           << "#define " << prefix << "_STRIDE_Y " << cell.getStrideY() << "\n"
           << "#define " << prefix << "_PADDING_X " << padding[0] << "\n"
           << "#define " << prefix << "_PADDING_Y " << padding[1] << "\n\n";


    C_CellExport::generateActivation(cell, header);
    C_CellExport::generateActivationScaling(cell, header);
}

void N2D2::C_ConvCellExport::generateHeaderFreeParameters(const ConvCell& cell,
                                                          std::ofstream& header)
{
    generateHeaderBias(cell, header);
    generateHeaderWeights(cell, header);
}

void N2D2::C_ConvCellExport::generateHeaderBias(const ConvCell& cell,
                                                std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
}

void N2D2::C_ConvCellExport::generateHeaderBiasVariable(const ConvCell& cell,
                                                        std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    header << "static const BDATA_T " << identifier << "_biases["
           << Utils::upperCase(identifier) << "_NB_OUTPUTS] = ";
}

void N2D2::C_ConvCellExport::generateHeaderBiasValues(const ConvCell& cell,
                                                      std::ofstream& header)
{
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    Tensor<Float_T> bias;

    header << "{";
    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if(output > 0)
            header << ", ";
            
        if (cell.getParameter<bool>("NoBias")) {
            header << "0";
        }
        else {
            cell.getBias(output, bias);

            CellExport::generateFreeParameter(bias(0), header);
        }

        CellExport::generateSingleShiftHalfAddition(cellFrame, output, header);
    }
    header << "};\n";
}

void N2D2::C_ConvCellExport::generateHeaderWeights(const ConvCell& cell,
                                                   std::ofstream& header)
{
    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        for (std::size_t channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (!cell.isConnection(channel, output))
                continue;

            generateHeaderKernelWeightsVariable(cell, header, output, channel);
            generateHeaderKernelWeightsValues(cell, header, output, channel);
        }
    }

    generateHeaderWeightsVariable(cell, header);
    generateHeaderWeightsValues(cell, header);
}

void N2D2::C_ConvCellExport::generateHeaderKernelWeightsVariable(
    const ConvCell& cell,
    std::ofstream& header,
    unsigned int output,
    unsigned int channel)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "static const WDATA_T " << identifier << "_weights_" << output
           << "_" << channel << "[" << prefix << "_KERNEL_HEIGHT][" << prefix
           << "_KERNEL_WIDTH] = ";
}

void
N2D2::C_ConvCellExport::generateHeaderKernelWeightsValues(const ConvCell& cell,
                                                          std::ofstream& header,
                                                          unsigned int output,
                                                          unsigned int channel)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    Tensor<Float_T> kernel;
    cell.getWeight(output, channel, kernel);

    header << "{\n";

    for (std::size_t sy = 0; sy < cell.getKernelHeight(); ++sy) {
        if (sy > 0)
            header << ",\n";

        header << "    {";

        for (std::size_t sx = 0; sx < cell.getKernelWidth(); ++sx) {
            if (sx > 0)
                header << ", ";

            CellExport::generateFreeParameter(kernel(sx, sy), header);
        }

        header << "}";
    }

    header << "};\n";
}

void N2D2::C_ConvCellExport::generateHeaderWeightsVariable(const ConvCell& cell,
                                                           std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "typedef WDATA_T " << prefix << "_KERNEL_T[" << prefix
           << "_KERNEL_HEIGHT][" << prefix << "_KERNEL_WIDTH];\n";
    header << "static const " << prefix << "_KERNEL_T* " << identifier
           << "_weights"
              "[" << prefix << "_NB_OUTPUTS][" << prefix << "_NB_CHANNELS] = ";
}

void N2D2::C_ConvCellExport::generateHeaderWeightsValues(const ConvCell& cell,
                                                         std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "{\n";

    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ",\n";

        header << "    {";

        for (std::size_t channel = 0; channel < cell.getNbChannels();
             ++channel) {
            if (channel > 0)
                header << ", ";

            if (!cell.isConnection(channel, output))
                header << "NULL";
            else
                header << "&" << identifier << "_weights_" << output << "_"
                       << channel;
        }

        header << "}";
    }

    header << "};\n\n";
}

std::unique_ptr<N2D2::C_ConvCellExport>
N2D2::C_ConvCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<C_ConvCellExport>(new C_ConvCellExport);
}

void N2D2::C_ConvCellExport::generateCellData(Cell& cell,
                                              const std::string& outputName,
                                              const std::string& outputSizeName,
                                              std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "static DATA_T " << outputName << "[" << outputSizeName << "]["
         << prefix << "_OUTPUTS_HEIGHT][" << prefix << "_OUTPUTS_WIDTH];\n";
}

void N2D2::C_ConvCellExport::generateCellFunction(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& /*parentCells*/,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& outputSizeName,
    std::ofstream& prog,
    bool isUnsigned,
    const std::string& funcProto,
    const std::string& memProto,
    bool memCompact)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto = (funcProto.empty()) ? "convcell" : funcProto;

    // Time analysis (start)
    prog << "#ifdef TIME_ANALYSIS\n"
        "   gettimeofday(&start, NULL);\n"
        "#endif\n";

    prog << "    " << proto << "_" << ((isUnsigned) ? "u" : "") << "propagate("
         << prefix << "_NB_CHANNELS, "
         << prefix << "_CHANNELS_HEIGHT, "
         << prefix << "_CHANNELS_WIDTH, "
         << prefix << "_PADDING_Y, "
         << prefix << "_PADDING_X, "
         << prefix << "_STRIDE_Y, "
         << prefix << "_STRIDE_X, "
         << prefix << "_SUB_SAMPLE_Y, "
         << prefix << "_SUB_SAMPLE_X, "
         << inputName << ", "
         << prefix << "_OY_SIZE, "
         << prefix << "_OX_SIZE, "
         << outputSizeName << ", "
         << prefix << "_OUTPUTS_HEIGHT, "
         << prefix << "_OUTPUTS_WIDTH, "
         << prefix << "_NB_OUTPUTS, "
         << prefix << "_OUTPUT_OFFSET, "
         << outputName << ", "
         << prefix << "_KERNEL_HEIGHT, "
         << prefix << "_KERNEL_WIDTH, "
         << memProto << identifier << "_biases, ";

    if (memCompact) {
        prog << memProto << identifier << "_weights_compact_map, "
             << memProto << identifier << "_weights_compact, ";
    } else
        prog << memProto << identifier << "_weights, ";

    prog << prefix << "_ACTIVATION, "
        << prefix << "_SHIFT);\n";

    // Time analysis (end)
    prog << "#ifdef TIME_ANALYSIS\n"
        "    gettimeofday(&end, NULL);\n"
        "    static RUNNING_MEAN_T " << identifier << "_timing = {0.0, 0};\n"
        "    time_analysis(\"" << identifier << "\", start, end, &"
        << identifier << "_timing);\n"
        "#endif\n";

    // Accumulation analysis
    prog << "#ifdef ACC_DYN_ANALYSIS\n"
        "    static SUM_T " << identifier << "_acc_min = 0;\n"
        "    static SUM_T " << identifier << "_acc_max = 0;\n"
        "    static SUM_T " << identifier << "_presat_min = 0;\n"
        "    static SUM_T " << identifier << "_presat_max = 0;\n"
        "    convcell_propagate_accs_report("
        "\"" << identifier << "\", "
        << prefix << "_NB_CHANNELS, "
        << prefix << "_CHANNELS_HEIGHT, "
        << prefix << "_CHANNELS_WIDTH, "
        << prefix << "_PADDING_Y, "
        << prefix << "_PADDING_X, "
        << prefix << "_STRIDE_Y, "
        << prefix << "_STRIDE_X, "
        << prefix << "_SUB_SAMPLE_Y, "
        << prefix << "_SUB_SAMPLE_X, "
        << inputName << ", "
        << prefix << "_OY_SIZE, "
        << prefix << "_OX_SIZE, "
        << prefix << "_OUTPUTS_HEIGHT, "
        << prefix << "_OUTPUTS_WIDTH, "
        << prefix << "_NB_OUTPUTS, "
        "&" << identifier << "_acc_min, "
        "&" << identifier << "_acc_max, "
        "&" << identifier << "_presat_min, "
        "&" << identifier << "_presat_max, "
        << prefix << "_KERNEL_HEIGHT, "
        << prefix << "_KERNEL_WIDTH, "
        << memProto << identifier << "_biases, "
        << memProto << identifier << "_weights, "
        "ACC_DYN_REPORT);\n"
        "#endif\n";

    // Dynamic data analysis
    prog << "#ifdef DATA_DYN_ANALYSIS\n"
        "    static DATA_T " << identifier << "_min = DATA_T_MAX;\n"
        "    static DATA_T " << identifier << "_max = DATA_T_MIN;\n"
        "    static RUNNING_MEAN_T " << identifier << "_mean = {0.0, 0};\n"
        "    convcell_outputs_dynamic_print("
        "\"" << identifier << "\", "
        << prefix << "_NB_OUTPUTS, "
        << prefix << "_OUTPUTS_HEIGHT, "
        << prefix << "_OUTPUTS_WIDTH, "
        << outputName << ", "
        << "&" << identifier << "_min, "
        << "&" << identifier << "_max, "
        << "&" << identifier << "_mean);\n"
        "#endif\n";

    // Save outputs
    prog << "#ifdef SAVE_OUTPUTS\n"
         << "    convcell_outputs_save("
            << "\"" << identifier << ".txt\", "
            << DeepNetExport::isCellOutputUnsigned(cell) << ","
            << prefix << "_NB_OUTPUTS, "
            << prefix << "_OUTPUTS_HEIGHT, "
            << prefix << "_OUTPUTS_WIDTH, "
            << outputName
         << ");\n"
         << "#endif\n";
}

void N2D2::C_ConvCellExport::generateOutputFunction(Cell& cell,
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
