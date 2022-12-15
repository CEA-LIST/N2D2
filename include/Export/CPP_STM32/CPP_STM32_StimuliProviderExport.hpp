/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifndef N2D2_CPP_STM32_STIMULI_PROVIDER_EXPORT_H
#define N2D2_CPP_STM32_STIMULI_PROVIDER_EXPORT_H

#include <string>

#include "Database/Database.hpp"
#include "Export/StimuliProviderExport.hpp"

namespace N2D2 {

class DeepNet;
class StimuliProvider;

class CPP_STM32_StimuliProviderExport: private StimuliProviderExport {
public:
    static void generate(const DeepNet& deepNet, StimuliProvider& sp,
                         const std::string& dirName,
                         Database::StimuliSet set,
                         bool unsignedData,
                         CellExport::Precision precision,
                         int nbStimuliMax);
};

}

#endif