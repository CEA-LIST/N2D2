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

#include "Export/CellExport.hpp"

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

long long int N2D2::CellExport::getIntFreeParameter(Cell& cell, double value)
{
    const double scaling = (double)(std::pow(2, mPrecision - 1) - 1);
    const bool sat = (std::abs(value) > 1.0);

    if (sat) {
        value = Utils::clamp(value, -1.0, 1.0);
        std::cout << Utils::cwarning
                  << "Warning: free parameter saturation in \""
                  << cell.getName() << "\"" << Utils::cdef << std::endl;
    }

    return getIntApprox(scaling * value, mIntApprox);
}

bool N2D2::CellExport::generateFreeParameter(Cell& cell,
                                             double value,
                                             std::ostream& stream)
{
    if (mPrecision > 0) {
        const double scaling = (double)(std::pow(2, mPrecision - 1) - 1);
        const bool sat = (std::abs(value) > 1.0);
        const double val = scaling * value;

        stream << getIntFreeParameter(cell, value);

        if (sat)
            stream << " /*SAT(" << val << ")*/";

        return sat;
    } else {
        if (mPrecision == Float64)
            stream << std::showpoint
                   << std::setprecision(std::numeric_limits<double>::digits10
                                        + 1) << ((double)value);
        else
            stream << std::showpoint << std::setprecision(std::numeric_limits
                                                          <float>::digits10 + 1)
                   << ((float)value) << "f";

        return false;
    }
}
