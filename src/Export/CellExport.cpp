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

#include "Activation/ActivationScalingMode.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/CellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "utils/Utils.hpp"

N2D2::CellExport::Precision N2D2::CellExport::mPrecision = Int8;
N2D2::CellExport::IntApprox N2D2::CellExport::mIntApprox = Round;
bool N2D2::CellExport::mWarnSat = true;

void N2D2::CellExport::generate(Cell& cell,
                                const std::string& dirName,
                                const std::string& type)
{
    const std::string cellType = cell.getType();

    if (Registrar<CellExport>::exists(cellType))
        Registrar<CellExport>::create(cellType)(cell, dirName, type);
    else {
        std::cout << Utils::cwarning << "Error: \"" << cellType << "\" cell"
            " type is not exportable for \"" << type << "\" export (if not used"
            " for inference, consider removing it before export)"
            << Utils::cdef << std::endl;
    }
}

long long int N2D2::CellExport::getIntApprox(double value, IntApprox method) {
    if (method == Floor)
        return std::floor(value);
    else if (method == Ceil)
        return std::ceil(value);
    else if (method == Truncate)
        return (long long int)value;
    else if (method == Round)
        return Utils::round(value);
    else if (method == PowerOfTwo) {
        const int sign = (value >= 0) ? 1 : -1;
        return sign
               * std::pow(
                     2, std::max(0, (int)Utils::round(log2(std::abs(value)))));
    }

    return 0;
}

long long int N2D2::CellExport::getIntFreeParameter(const Cell& cell, double value, 
                                                    Cell::FreeParametersType freeParameterType)
{
    const bool sat = (freeParameterType == Cell::Additive)?false:(std::abs(value) > 1.0);
    if (sat) {
        value = Utils::clamp(value, -1.0, 1.0);
        std::cout << Utils::cwarning
                  << "Warning: free parameter saturation in \""
                  << cell.getName() << "\"" << Utils::cdef << std::endl;

        if (mWarnSat) {
            std::cout << "Saturation detected! This may lead to poor network"
                " performance...\n"
                "Consider using normalized free parameters for ReLU-only"
                " activations\n"
                "  (-w weights_normalized).\n"
                "Continue anyway (will not stop again)? (y/n) ";

            std::string cmd;

            do {
                std::cin >> cmd;
            }
            while (cmd != "y" && cmd != "n");

            if (cmd == "n") {
                std::cout << "Exiting..." << std::endl;
                std::exit(0);
            }

            mWarnSat = false;
        }
    }
    
    const double scaling = getScalingForFreeParameterType(cell, freeParameterType);
    return getIntApprox(scaling * value, mIntApprox);
}

bool N2D2::CellExport::generateFreeParameter(const Cell& cell, double value, std::ostream& stream,
                                             Cell::FreeParametersType freeParameterType,
                                             bool typeAccuracy)
{
    if (mPrecision > 0) {
        stream << getIntFreeParameter(cell, value, freeParameterType);

        const bool sat = (freeParameterType == Cell::Additive)?false:(std::abs(value) > 1.0);
        if (sat) {
            const double val = getScalingForFreeParameterType(cell, freeParameterType) * value;
            stream << " /*SAT(" << val << ")*/";
        }

        return sat;
    } else {
        if (mPrecision == Float64)
            stream << std::showpoint
                   << std::setprecision(std::numeric_limits<double>::digits10
                                        + 1) << ((double)value);
        else {
            stream << std::showpoint << std::setprecision(std::numeric_limits
                                                          <float>::digits10 + 1)
                   << ((float)value);

            if(typeAccuracy)
                stream << "f";
        }
        return false;
    }
}

double N2D2::CellExport::getScalingForFreeParameterType(const Cell& cell, Cell::FreeParametersType freeParameterType) {
    double scaling = (double) std::pow(2, mPrecision - 1) - 1;

    // For the bias we also need to scale it by the maximum value of the input type.
    // A bias is just like an extra connection where the input is equal to 1.0.
    if(freeParameterType == Cell::Additive) {
        if(DeepNetExport::isCellInputsUnsigned(cell)) {
            scaling *= (std::pow(2, (int) mPrecision) - 1);
        }
        else {
            scaling *= (std::pow(2, (int) mPrecision - 1) - 1);
        }
    }

    return scaling;
}


void N2D2::CellExport::generateShiftScalingHalfAddition(const Cell_Frame_Top& cellFrame, std::size_t output, 
                                                        std::ostream& stream)
{
    if(cellFrame.getActivation()->getActivationScaling().getMode() == ActivationScalingMode::SINGLE_SHIFT) {
        const std::size_t shift = cellFrame.getActivation()->getActivationScaling()
                                                            .getSingleShiftScaling()
                                                            .getScalingPerOutput()[output];
        const std::size_t half = 1 << (shift - 1);

        stream << " + " << half;
    }
    else if(cellFrame.getActivation()->getActivationScaling().getMode() == ActivationScalingMode::DOUBLE_SHIFT) {
        const std::pair<unsigned char, unsigned char> shift = cellFrame.getActivation()
                                                                      ->getActivationScaling()
                                                                       .getDoubleShiftScaling()
                                                                       .getScalingPerOutput()[output];
        
        const std::size_t half = 1 << (shift.first - 1);

        if(shift.second == DoubleShiftScaling::NO_SHIFT) {
            stream << " + " << half;
        }
        else {
            stream << " + " << static_cast<std::size_t>(
                                   std::ceil(1.0*half*std::pow(2, shift.second)/
                                                     (std::pow(2, shift.second) + 1))
                               );
        }
    }
}