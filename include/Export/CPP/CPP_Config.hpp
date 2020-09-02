/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/
#ifndef N2D2_CPP_CONFIG_H
#define N2D2_CPP_CONFIG_H

#include <string>

#include "Export/MemoryManager.hpp"

namespace N2D2 {

class CPP_Config {
public:
    static const std::string INCLUDE_INPUT_IN_BUFFER;
    static const bool INCLUDE_INPUT_IN_BUFFER_DEFAULT;

    static const std::string OPTIMIZE_BUFFER_MEMORY;
    static const bool OPTIMIZE_BUFFER_MEMORY_DEFAULT;

    static const std::string OPTIMIZE_NOBRANCH_CONCAT;
    static const bool OPTIMIZE_NOBRANCH_CONCAT_DEFAULT;

    static const std::string MEMORY_ALIGNMENT;
    static const int MEMORY_ALIGNMENT_DEFAULT;

    static const std::string MEMORY_MANAGER_STRATEGY;
    static const MemoryManager::OptimizeStrategy MEMORY_MANAGER_STRATEGY_DEFAULT;

};
}

#endif