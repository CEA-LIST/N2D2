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

#ifndef N2D2_TRANSFORMATIONGENERATOR_H
#define N2D2_TRANSFORMATIONGENERATOR_H

#include <memory>

#include "Transformation/Transformation.hpp"
#include "utils/IniParser.hpp"
#include "utils/Registrar.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@AffineTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ApodizationTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ChannelExtractionTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ColorSpaceTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@DFTTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FilterTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FlipTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@MagnitudePhaseTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@NormalizeTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@PadCropTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@RangeAffineTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@RescaleTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ReshapeTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ThresholdTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@TrimTransformationGenerator@N2D2@@0U?$Registrar@VTransformationGenerator@N2D2@@@2@A")
#endif

namespace N2D2 {
class TransformationGenerator {
public:
    typedef std::function<std::shared_ptr<Transformation>(
        IniParser& iniConfig, const std::string& section)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static std::shared_ptr<Transformation>
    generate(IniParser& iniConfig,
             const std::string& section,
             const std::string& typePropertyName = "Type");
};
}

#endif // N2D2_TRANSFORMATIONGENERATOR_H
