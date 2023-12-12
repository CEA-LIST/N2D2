/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <string>
#include "Export/CPP/CPP_Config.hpp"


const std::string N2D2::CPP_Config::INCLUDE_INPUT_IN_BUFFER = "IncludeInputInBuffer";
const bool N2D2::CPP_Config::INCLUDE_INPUT_IN_BUFFER_DEFAULT = true;

const std::string N2D2::CPP_Config::OPTIMIZE_BUFFER_MEMORY = "OptimizeBufferMemory";
const bool N2D2::CPP_Config::OPTIMIZE_BUFFER_MEMORY_DEFAULT = true;

const std::string N2D2::CPP_Config::OPTIMIZE_NOBRANCH_CONCAT = "OptimizeNoBranchConcat";
const bool N2D2::CPP_Config::OPTIMIZE_NOBRANCH_CONCAT_DEFAULT = true;

const std::string N2D2::CPP_Config::MEMORY_ALIGNMENT = "MemoryAlignment";
const int N2D2::CPP_Config::MEMORY_ALIGNMENT_DEFAULT = 1;

const std::string N2D2::CPP_Config::MEMORY_MANAGER_STRATEGY = "MemoryManagerStrategy";
const N2D2::MemoryManager::OptimizeStrategy N2D2::CPP_Config::MEMORY_MANAGER_STRATEGY_DEFAULT = N2D2::MemoryManager::OptimizeMaxLifetimeMaxSizeFirst;

const std::string N2D2::CPP_Config::DILATED_WEIGHTS = "DilatedWeights";
const bool N2D2::CPP_Config::DILATED_WEIGHTS_DEFAULT = false;
