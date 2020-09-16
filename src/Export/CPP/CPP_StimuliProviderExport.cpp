/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <string>

#include "DeepNet.hpp"
#include "StimuliProvider.hpp"
#include "Database/Database.hpp"
#include "Export/CPP/CPP_StimuliProviderExport.hpp"
#include "Export/StimuliProviderExport.hpp"
#include "utils/Registrar.hpp"

N2D2::Registrar<N2D2::StimuliProviderExport>
N2D2::CPP_StimuliProviderExport::mRegistrar(
    {"CPP", "CPP_ASMP", "CPP_HLS"},
    N2D2::CPP_StimuliProviderExport::generate);

void N2D2::CPP_StimuliProviderExport::generate(const DeepNet& deepNet,
                                               StimuliProvider& sp,
                                               const std::string& dirName,
                                               Database::StimuliSet set,
                                               bool unsignedData,
                                               CellExport::Precision precision,
                                               int nbStimuliMax)
{
    StimuliProviderExport::generate(deepNet, sp, dirName, set,
        unsignedData, precision, nbStimuliMax, StimuliProviderExport::HWC);
}
