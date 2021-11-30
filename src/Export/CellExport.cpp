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
#include "DeepNet.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/CellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "utils/Utils.hpp"

N2D2::CellExport::Precision N2D2::CellExport::mPrecision = Int8;
N2D2::CellExport::IntApprox N2D2::CellExport::mIntApprox = Round;

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

long long int N2D2::CellExport::getIntFreeParameter(double value, bool truncMode) {
    if(!Utils::isIntegral(value, 0.01) && !truncMode) {
#pragma omp critical
        throw std::runtime_error("Can't export a non-integral floating-point as an integral parameter. "
                                 "The network must be quantized beforehand with the -calib option "
                                 "if an integer export is necessary.");
    }

    return static_cast<long long int>(std::round(value));
}

void N2D2::CellExport::generateFreeParameter(double value, std::ostream& stream,
                                             bool typeAccuracy)
{
    if (mPrecision > 0) {
        stream << getIntFreeParameter(value);
    } 
    else {
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
    }
}

void N2D2::CellExport::generateSingleShiftHalfAddition(const Cell_Frame_Top& cellFrame, std::size_t output, 
                                                       std::ostream& stream)
{
    if (!cellFrame.getActivation())
        return;

    if(cellFrame.getActivation()->getActivationScaling().getMode() == ScalingMode::SINGLE_SHIFT) {
        const std::size_t shift = cellFrame.getActivation()->getActivationScaling()
                                                            .getSingleShiftScaling()
                                                            .getScalingPerOutput()[output];

        if (shift > 0) {
            const std::size_t half = static_cast<std::size_t>(1) << (shift - 1);

            stream << " + " << half;
        }
    }
}

int N2D2::CellExport::getQuantizedOutputNbBits(const DeepNet& deepNet,
                                               const Cell& cell)
{
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    const std::shared_ptr<N2D2::Activation> activation
        = cellFrame.getActivation();

    if (activation) {
        return (activation->isQuantized())
            ? activation->getQuantizedNbBits()          // QAT
            : (int)CellExport::mPrecision;              // Post-training
    }
    else {
        int nbBits = 0;
        const std::vector<std::shared_ptr<Cell> >& parents
            = deepNet.getParentCells(cell.getName());

        for (const std::shared_ptr<Cell>& parent: parents) {
            const int parentNbBits = (parent)
                ? getQuantizedOutputNbBits(deepNet, *parent)
                : (int)CellExport::mPrecision;

            if (nbBits == 0)
                nbBits = parentNbBits;
            else if (parentNbBits != nbBits) {
                std::stringstream msgStr;
                msgStr << "Inputs precision mismatch for cell \""
                    << cell.getName() << "\": first parent (\""
                    << ((parents[0]) ? parents[0]->getName() : "env")
                    << "\" is " << nbBits << ", whereas parent \""
                    << ((parent) ? parent->getName() : "env") << "\" is "
                    << parentNbBits << ")";

                throw std::runtime_error(msgStr.str());
            }
        }

        return nbBits;
    }
}
