/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/
#include "env.hpp"
#include "network.h"
#include "Network.hpp"

N2D2_Export::Network network;

void propagate(const void* inputs, int32_t* outputs) {
#if ENV_DATA_UNSIGNED == 1
    network.propagate((const UDATA_T*) inputs, outputs);
#else
    network.propagate((const DATA_T*) inputs, outputs);
#endif
}
