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
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"


static const N2D2::Registrar<N2D2::ScalingCellExport> registrar("CPP",  
                                                                N2D2::CPP_ScalingCellExport::generate);

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
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n"
           << "\n";
    
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
        header << "static const std::int32_t " << prefix << "_SCALING_FACTOR_PER_OUTPUT[] = {"
               << Utils::join(fpScaling.getScalingPerOutput().begin(), 
                              fpScaling.getScalingPerOutput().end(), ',') 
               << "};\n";
    }
    else {
        throw std::runtime_error("Unsupported scaling mode for cell " + cell.getName() + ".");
    }

    
    header << "\n";
}
