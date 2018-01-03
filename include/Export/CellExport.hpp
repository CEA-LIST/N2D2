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

#ifndef N2D2_CELLEXPORT_H
#define N2D2_CELLEXPORT_H

#include "Cell/Cell.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@BatchNormCellExport@N2D2@@0U?$Registrar@VCellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ConvCellExport@N2D2@@0U?$Registrar@VCellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FMPCellExport@N2D2@@0U?$Registrar@VCellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FcCellExport@N2D2@@0U?$Registrar@VCellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@PoolCellExport@N2D2@@0U?$Registrar@VCellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@SoftmaxCellExport@N2D2@@0U?$Registrar@VCellExport@N2D2@@@2@A")
#endif

namespace N2D2 {
/**
 * Base class for methods commun to every cell type for any export type
 * ANY CELL, ANY EXPORT
*/
class CellExport {
public:
    enum Precision {
        Int1 = 1,
        Int2 = 2,
        Int3 = 3,
        Int4 = 4,
        Int5 = 5,
        Int6 = 6,
        Int7 = 7,
        Int8 = 8,
        Int9 = 9,
        Int10 = 10,
        Int11 = 11,
        Int12 = 12,
        Int13 = 13,
        Int14 = 14,
        Int15 = 15,
        Int16 = 16,
        Int32 = 32,
        Int64 = 64,
        Float16 = -16,
        Float32 = -32,
        Float64 = -64
    };

    enum IntApprox {
        Floor,
        Ceil,
        Round,
        PowerOfTwo
    };

    typedef std::function
        <void(Cell&, const std::string&, const std::string&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static Precision mPrecision;
    static IntApprox mIntApprox;

    static void
    generate(Cell& cell, const std::string& dirName, const std::string& type);

    static bool
    generateFreeParameter(Cell& cell, double value, std::ostream& stream);
    static long long int getIntApprox(double value, IntApprox method = Round);
    static long long int getIntFreeParameter(Cell& cell, double value);
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::CellExport::IntApprox>::data[]
    = {"Floor", "Ceil", "Round", "PowerOfTwo"};
}

#endif // N2D2_CELLEXPORT_H
