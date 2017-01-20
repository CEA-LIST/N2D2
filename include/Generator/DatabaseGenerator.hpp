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

#ifndef N2D2_DATABASEGENERATOR_H
#define N2D2_DATABASEGENERATOR_H

#include <memory>

#include "Database/Database.hpp"
#include "utils/IniParser.hpp"
#include "utils/Registrar.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@CIFAR100_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@CIFAR10_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@CKP_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@Caltech101_DIR_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@Caltech256_DIR_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@CaltechPedestrian_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@DIR_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FDDB_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@GTSDB_DIR_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@GTSRB_DIR_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ILSVRC2012_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@LITISRouen_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@MNIST_IDX_DatabaseGenerator@N2D2@@0U?$Registrar@VDatabaseGenerator@N2D2@@@2@A")
#endif

namespace N2D2 {
class DatabaseGenerator {
public:
    typedef std::function<std::shared_ptr<Database>(
        IniParser& iniConfig, const std::string& section)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static std::shared_ptr<Database> generate(IniParser& iniConfig,
                                              const std::string& section);
};
}

#endif // N2D2_DATABASEGENERATOR_H
