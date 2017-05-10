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

#ifndef N2D2_CELLGENERATOR_H
#define N2D2_CELLGENERATOR_H

#include "Cell/Cell.hpp"
#include "Generator/ActivationGenerator.hpp"
#include "Generator/SolverGenerator.hpp"
#include "Generator/TanhActivationGenerator.hpp"
#include "utils/IniParser.hpp"
#include "utils/Registrar.hpp"
#include "DeepNet.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@AnchorCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@BatchNormCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ConvCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@DeconvCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@DropoutCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FMPCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FcCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@LRNCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@PoolCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ROIPoolingCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@RPCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@SoftmaxCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@TransformationCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@UnpoolCellGenerator@N2D2@@0U?$Registrar@VCellGenerator@N2D2@@@2@A")
#endif

namespace N2D2 {
class CellGenerator {
public:
    typedef std::function<std::shared_ptr<Cell>(
        Network& network,
        StimuliProvider& sp,
        const std::vector<std::shared_ptr<Cell> >& parents,
        IniParser& iniConfig,
        const std::string& section)> RegistryCreate_T;
    typedef std::function<void(
        const std::shared_ptr<Cell>& cell,
        const std::shared_ptr<DeepNet>& deepNet,
        IniParser& iniConfig,
        const std::string& section)> RegistryPostCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static std::string mDefaultModel;

    static std::shared_ptr<Cell> generate(Network& network,
                                          StimuliProvider& sp,
                                          const std::vector
                                          <std::shared_ptr<Cell> >& parents,
                                          IniParser& iniConfig,
                                          const std::string& section);
    static void postGenerate(const std::shared_ptr<Cell>& cell,
                             const std::shared_ptr<DeepNet>& deepNet,
                             IniParser& iniConfig,
                             const std::string& section);

protected:
    static std::map<std::string, std::string>
    getConfig(const std::string& model, IniParser& iniConfig);
};
}

#endif // N2D2_CELLGENERATOR_H
