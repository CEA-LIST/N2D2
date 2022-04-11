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

#include "ScalingMode.hpp"
#include "Activation/Activation.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/C/C_CellExport.hpp"
#include "Cell/ScalingCell.hpp"
#include "Cell/ElemWiseCell.hpp"

void N2D2::C_CellExport::generateHeaderBegin(const Cell& cell, std::ofstream& header) {
    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    header << "// N2D2 auto-generated file.\n"
              "// @ " << std::asctime(localNow)
           << "\n"; // std::asctime() already appends end of line

    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                cell.getName()));

    header << "#ifndef N2D2_EXPORTC_" << prefix << "_LAYER_H\n"
                                                   "#define N2D2_EXPORTC_"
           << prefix << "_LAYER_H\n\n";
}

void N2D2::C_CellExport::generateHeaderIncludes(const Cell& /*cell*/, std::ofstream& header) {
    header << "#include \"typedefs.h\"\n";
}

void N2D2::C_CellExport::generateHeaderEnd(const Cell& cell, std::ofstream& header) {
    header << "#endif // N2D2_EXPORTC_"
           << Utils::upperCase(Utils::CIdentifier(cell.getName()))
           << "_LAYER_H" << std::endl;
    header.close();
}

void N2D2::C_CellExport::generateActivation(const Cell& cell, std::ofstream& header) {
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    const std::string type = (cellFrame.getActivation())
        ? cellFrame.getActivation()->getType() : "Linear";

    header << "#define " << prefix << "_ACTIVATION " << type << "\n";

    if (cellFrame.getActivation()
        && (type == "Rectifier" || type == "Linear"))
    {
        const double clipping = cellFrame.getActivation()
            ->getParameter<double>("Clipping");

        if (clipping > 0) {
            std::cout << Utils::cwarning << "Clipping (" << clipping << ") for "
                << type << " in cell " << cell.getName() << " is not supported!"
                << Utils::cdef << std::endl;
        }
    }
}

void N2D2::C_CellExport::generateActivationScaling(const Cell& cell, std::ofstream& header) {
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);


    if (cellFrame.getActivation() == nullptr) {
        header << "#define " << prefix << "_SHIFT 0\n";
        header << "static const int32_t " << prefix << "_SCALING_FACTOR_PER_OUTPUT[] = {0};\n";
        return;
    }
    
    const Activation& activation = *cellFrame.getActivation();
    if(activation.getActivationScaling().getMode() == ScalingMode::NONE) {
        header << "#define " << prefix << "_SHIFT 0\n";
        header << "static const int32_t " << prefix << "_SCALING_FACTOR_PER_OUTPUT[] = {0};\n";
    }
    else if(activation.getActivationScaling().getMode() == ScalingMode::SINGLE_SHIFT) {
        const std::vector<unsigned char>& scaling = activation.getActivationScaling()
                                                              .getSingleShiftScaling()
                                                              .getScalingPerOutput();
        if(!Utils::all_same(scaling.begin(), scaling.end())) {
            throw std::runtime_error("Single-shift with a global scaling per layer is the only activation "
                                     "scaling mode supported by the export.");
        }

        header << "#define " << prefix << "_SHIFT " << +scaling.front() << "\n";
        header << "static const int32_t " << prefix << "_SCALING_FACTOR_PER_OUTPUT[] = {0};\n";
    }
    else if(activation.getActivationScaling().getMode() == ScalingMode::FIXED_MULT16
        || activation.getActivationScaling().getMode() == ScalingMode::FIXED_MULT32)
    {
        if (cell.getType() == ScalingCell::Type || cell.getType() == ElemWiseCell::Type) {
            const FixedPointScaling& fpScaling = activation.getActivationScaling().getFixedPointScaling();

            header << "#define " << prefix << "_SHIFT " << fpScaling.getFractionalBits() << "\n";
            header << "static const int32_t " << prefix << "_SCALING_FACTOR_PER_OUTPUT[] = {"
                   << Utils::join(fpScaling.getScalingPerOutput().begin(), 
                                  fpScaling.getScalingPerOutput().end(), ',') 
                   << "};\n";
        }
        else {
            throw std::runtime_error("Scaling and ElemWise are the only layers "
                                     "supported in Fixed-point scaling mode.");
        }
    }
    else {
        throw std::runtime_error("Single-shift with a global scaling per layer is the only activation "
                                 "scaling mode supported by the export.");
    }

    header << "\n";
} 
