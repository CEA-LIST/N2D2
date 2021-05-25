/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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
#include "Export/C/C_CellExport.hpp"
#include "Export/C/C_ScalingCellExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

N2D2::Registrar<N2D2::ScalingCellExport>
N2D2::C_ScalingCellExport::mRegistrar("C", N2D2::C_ScalingCellExport::generate);

N2D2::Registrar<N2D2::C_CellExport>
N2D2::C_ScalingCellExport::mRegistrarType(  ScalingCell::Type,
                                            N2D2::C_ScalingCellExport::getInstance);

void N2D2::C_ScalingCellExport::generate(const ScalingCell& cell, const std::string& dirName) {
    Utils::createDirectories(dirName + "/include");

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    const std::string fileName = dirName + "/include/" + identifier + ".h";
    std::ofstream header(fileName);
    C_CellExport::generateHeaderBegin(cell, header);
    C_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    C_CellExport::generateHeaderEnd(cell, header);

    if(!header) {
        throw std::runtime_error("Error writing file: " + fileName);
    }
}

void N2D2::C_ScalingCellExport::generateHeaderConstants(const ScalingCell& cell, std::ofstream& header) {
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n"
           << "\n";

    generateScaling(cell, header);
}

void N2D2::C_ScalingCellExport::generateScaling(const ScalingCell& cell, std::ofstream& header) {
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    if(cell.getScaling().getMode() == ScalingMode::FLOAT_MULT) {
        const std::vector<Float_T>& rescaleFactorPerOutput = cell.getScaling()
                                                                 .getFloatingPointScaling()
                                                                 .getScalingPerOutput();

        header << "static const float " << prefix << "_SCALING_FACTOR_PER_OUTPUT[] = {"
               << Utils::join(rescaleFactorPerOutput.begin(), rescaleFactorPerOutput.end(), ',') 
               << "};\n";
    }
    else if(cell.getScaling().getMode() == ScalingMode::FIXED_MULT) {
        const FixedPointScaling& fpScaling = cell.getScaling().getFixedPointScaling();

        header << "#define " << prefix << "_NB_FRACTIONAL_BITS " << fpScaling.getFractionalBits() << "\n";
        header << "static const int32_t " << prefix << "_SCALING_FACTOR_PER_OUTPUT[] = {"
               << Utils::join(fpScaling.getScalingPerOutput().begin(), 
                              fpScaling.getScalingPerOutput().end(), ',') 
               << "};\n";
    }
    else {
        throw std::runtime_error("Unsupported scaling mode for cell " + cell.getName() + ".");
    }
    
    header << "\n";
}

std::unique_ptr<N2D2::C_ScalingCellExport>
N2D2::C_ScalingCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<C_ScalingCellExport>(new C_ScalingCellExport);
}

void N2D2::C_ScalingCellExport::generateCellData(Cell& cell,
                                                   const std::string& outputName,
                                                   const std::string& outputSizeName,
                                                   std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "static DATA_T " << outputName << "[" << outputSizeName << "]["
         << prefix << "_OUTPUTS_HEIGHT][" << prefix << "_OUTPUTS_WIDTH];\n";
}

void N2D2::C_ScalingCellExport::generateCellFunction(
    Cell& cell,
    const std::vector<std::shared_ptr<Cell> >& /*parentCells*/,
    const std::string& inputName,
    const std::string& outputName,
    const std::string& /*outputSizeName*/,
    std::ofstream& prog,
    bool /*isUnsigned*/,
    const std::string& funcProto,
    const std::string& /*memProto*/,
    bool /*memCompact*/)
{

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const std::string proto = (funcProto.empty()) ? "scalingcell" : funcProto;

    prog << "    " << proto << "_" << "propagate("
        << prefix << "_NB_CHANNELS, "
        << prefix << "_CHANNELS_HEIGHT, "
        << prefix << "_CHANNELS_WIDTH, "
        << inputName << ", "
        << prefix << "_OUTPUTS_HEIGHT, "
        << prefix << "_OUTPUTS_WIDTH, "
        << prefix << "_NB_OUTPUTS, "
        << prefix << "_OUTPUT_OFFSET, "
        << outputName << ", "
        << prefix << "_SCALING_FACTOR_PER_OUTPUT, "
        << prefix << "_NB_FRACTIONAL_BITS);\n ";

    // Save outputs
    prog << "#ifdef SAVE_OUTPUTS\n"
         << "    scalingcell_outputs_save("
            << "\"" << identifier << ".txt\", "
            << DeepNetExport::isCellOutputUnsigned(cell) << ","
            << prefix << "_NB_OUTPUTS, "
            << prefix << "_OUTPUT_OFFSET, "
            << prefix << "_OUTPUTS_HEIGHT, "
            << prefix << "_OUTPUTS_WIDTH, "
            << outputName
         << ");\n"
         << "#endif\n";
}

void N2D2::C_ScalingCellExport::generateOutputFunction(Cell& cell,
                                                         const std::string& inputName,
                                                         const std::string& outputName,
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
