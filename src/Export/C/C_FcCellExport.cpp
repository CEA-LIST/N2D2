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

#include "Cell/ConvCell.hpp"
#include "Cell/PoolCell.hpp"
#include "Export/C/C_FcCellExport.hpp"

N2D2::Registrar<N2D2::FcCellExport>
N2D2::C_FcCellExport::mRegistrar("C", N2D2::C_FcCellExport::generate);

N2D2::Registrar<N2D2::C_CellExport>
N2D2::C_FcCellExport::mRegistrarType(FcCell::Type,
                                     N2D2::C_FcCellExport::getInstance);

void N2D2::C_FcCellExport::generate(const FcCell& cell, const std::string& dirName)
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

void N2D2::C_FcCellExport::generateHeaderConstants(const FcCell& cell,
                                                   std::ofstream& header)
{
    // Constants
    const std::size_t channelsSize = cell.getInputsSize();
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << channelsSize << "\n\n";

    C_CellExport::generateActivation(cell, header);
    C_CellExport::generateActivationScaling(cell, header);
}

void N2D2::C_FcCellExport::generateHeaderFreeParameters(const FcCell& cell,
                                                        std::ofstream& header)
{
    generateHeaderBias(cell, header);

    if (mThreshold > 0.0)
        generateHeaderWeightsSparse(cell, header);
    else
        generateHeaderWeights(cell, header);
}

void N2D2::C_FcCellExport::generateHeaderBias(const FcCell& cell,
                                              std::ofstream& header)
{
    generateHeaderBiasVariable(cell, header);
    generateHeaderBiasValues(cell, header);
}

void N2D2::C_FcCellExport::generateHeaderBiasVariable(const FcCell& cell,
                                                      std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    header << "static const BDATA_T " << identifier << "_biases["
           << Utils::upperCase(identifier) << "_NB_OUTPUTS] = ";
}

void N2D2::C_FcCellExport::generateHeaderBiasValues(const FcCell& cell,
                                                    std::ofstream& header)
{
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    Tensor<Float_T> bias;
    
    header << "{";
    for (std::size_t output = 0; output < cell.getNbOutputs(); output++) {
        if (cell.getParameter<bool>("NoBias")) {
            header << "0";
        }
        else {
            cell.getBias(output, bias);

            CellExport::generateFreeParameter(cell, bias(0), header, Cell::Additive);
            CellExport::generateSingleShiftHalfAddition(cellFrame, output, header);
        }

        header << ", ";
    }
    header << "};\n";
}

void N2D2::C_FcCellExport::generateHeaderWeights(const FcCell& cell,
                                                 std::ofstream& header)
{
    generateHeaderWeightsVariable(cell, header);
    generateHeaderWeightsValues(cell, header);
}

void N2D2::C_FcCellExport::generateHeaderWeightsVariable(const FcCell& cell,
                                                         std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "#define " << prefix << "_NB_WEIGHTS (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_NB_CHANNELS)\n\n";

    // Weights
    header << "static const WDATA_T " << identifier << "_weights[" << prefix
           << "_NB_OUTPUTS]"
              "[" << prefix << "_NB_CHANNELS] = \n";
}

void N2D2::C_FcCellExport::generateHeaderWeightsValues(const FcCell& cell,
                                                       std::ofstream& header)
{
    const std::size_t channelsSize = cell.getInputsSize();

    header << "{\n";
    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ",\n";

        header << "    {";

        for (std::size_t channel = 0; channel < channelsSize; ++channel) {
            if (channel > 0)
                header << ", ";

            Tensor<Float_T> weight;
            cell.getWeight(output, channel, weight);

            CellExport::generateFreeParameter(
                cell, weight(0), header, Cell::Multiplicative);
        }

        header << "}";
    }

    header << "};\n\n";
}

void N2D2::C_FcCellExport::generateHeaderWeightsSparse(const FcCell& cell,
                                                       std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::size_t channelsSize = cell.getInputsSize();

    std::vector<double> weights;
    std::vector<std::size_t> offsets;
    std::size_t offset = 0;

    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        for (std::size_t channel = 0; channel < channelsSize; ++channel) {
            Tensor<double> weight;
            cell.getWeight(output, channel, weight);

            if (std::fabs(weight(0)) >= mThreshold) {
                weights.push_back(weight(0));
                offsets.push_back(offset);
                offset = 1;
            } else
                ++offset;
        }
    }

    const std::size_t nbWeights = weights.size();

    header << "#define " << prefix << "_NB_WEIGHTS " << nbWeights << "\n"
           << "static WDATA_T " << identifier << "_weights_sparse["
           << prefix << "_NB_WEIGHTS] = {\n";

    for (std::size_t i = 0; i < nbWeights; ++i) {
        if (i > 0)
            header << ", ";

        CellExport::generateFreeParameter(cell, weights[i], header, Cell::Multiplicative);
    }

    header << "};\n\n";

    header << "static unsigned short " << identifier << "_weights_offsets["
           << prefix << "_NB_WEIGHTS] = {\n";

    for (std::size_t i = 0; i < nbWeights; ++i) {
        if (i > 0)
            header << ", ";

        header << offsets[i];
    }

    header << "};\n\n";

    std::cout << Utils::cnotice << "Sparse weights ratio: " << nbWeights << "/"
              << (cell.getNbOutputs() * channelsSize) << " ("
              << 100.0
                 * (nbWeights / (double)(cell.getNbOutputs() * channelsSize))
              << "%)" << Utils::cdef << std::endl;
}

std::unique_ptr<N2D2::C_FcCellExport>
N2D2::C_FcCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<C_FcCellExport>(new C_FcCellExport);
}

void N2D2::C_FcCellExport::generateCellData(Cell& /*cell*/,
                                            const std::string& outputName,
                                            const std::string& outputSizeName,
                                            std::ofstream& prog)
{
    prog << "static DATA_T " << outputName << "[" << outputSizeName << "];\n";
}

void N2D2::C_FcCellExport::generateCellFunction(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& parentCells,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& outputSizeName,
    std::ofstream& prog,
    bool isUnsigned,
    const std::string& funcProto,
    const std::string& memProto,
    bool /*memCompact*/)
{
    // funcProto and memProto parameters are extensions used in the C_HLS and
    // CPP_OpenCL exports
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto = (funcProto.empty()) ? "fccell" : funcProto;

    const bool input2d
        = (!parentCells[0]
           || parentCells[0]->getType()
              == ConvCell::Type /*|| parentCells[0]->getType() == "Lc"*/
           || parentCells[0]->getType() == PoolCell::Type);

    // Time analysis (start)
    prog << "#ifdef TIME_ANALYSIS\n"
        "   gettimeofday(&start, NULL);\n"
        "#endif\n";

    if (input2d) {
        prog << "    " << proto << "_" << ((isUnsigned) ? "u" : "")
             << "propagate_2d" << ((mThreshold > 0.0) ? "_sparse" : "") << "(";

        if (!parentCells[0]) {
            prog << "ENV_NB_OUTPUTS, "
                 << "ENV_SIZE_Y, "
                 << "ENV_SIZE_X, ";
        }
        else {
            const std::string prefixParent
                = Utils::upperCase(Utils::CIdentifier(
                                                parentCells[0]->getName()));

            std::stringstream prefixParentsCell;
            for (std::size_t i = 0; i < parentCells.size(); ++i)
                prefixParentsCell << Utils::upperCase(parentCells[i]->getName())
                                     + "_";

            prog << Utils::CIdentifier(prefixParentsCell.str())
                << "NB_OUTPUTS, "
                << prefixParent << "_OUTPUTS_HEIGHT, "
                << prefixParent << "_OUTPUTS_WIDTH, ";
        }
    }
    else {
        prog << "    " << proto << "_" << ((isUnsigned) ? "u" : "")
             << "propagate" << ((mThreshold > 0.0) ? "_sparse" : "") << "("
             << prefix << "_NB_CHANNELS, ";
    }

    prog << inputName << ", "
        << outputSizeName << ", "
        << prefix << "_NB_OUTPUTS, "
        << prefix << "_OUTPUT_OFFSET, "
        << outputName << ", ";

    if (input2d)
        prog << prefix << "_NB_CHANNELS, ";

    prog << memProto << identifier << "_biases, ";

    if (mThreshold > 0.0) {
        prog << memProto << prefix << "_NB_WEIGHTS, "
            << memProto << identifier << "_weights_sparse, "
            << memProto << identifier << "_weights_offsets, ";
    }
    else
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

    // Dynamic data analysis
    prog << "#ifdef DATA_DYN_ANALYSIS\n"
            "    static DATA_T " << identifier << "_min = DATA_T_MAX;\n"
            "    static DATA_T " << identifier << "_max = DATA_T_MIN;\n"
            "    static RUNNING_MEAN_T " << identifier << "_mean = {0.0, 0};\n"
            "    fccell_outputs_dynamic_print(\""
         << identifier << "\", " << prefix << "_NB_OUTPUTS, " << outputName
         << ", "
         << "&" << identifier << "_min, "
         << "&" << identifier << "_max, "
         << "&" << identifier << "_mean);\n"
            "#endif\n";

    // Save outputs
    prog << "#ifdef SAVE_OUTPUTS\n"
         << "    fccell_outputs_save("
            << "\"" << identifier << ".txt\", "
            << prefix << "_NB_OUTPUTS, "
            << outputName
         << ");\n"
         << "#endif\n";
}

void N2D2::C_FcCellExport::generateOutputFunction(Cell& cell,
                                                  const std::string& inputName,
                                                  const std::string& outputName,
                                                  std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "\n"
            "    output_max(" << prefix << "_NB_OUTPUTS, " << inputName << ", "
         << outputName << ");\n";
}
