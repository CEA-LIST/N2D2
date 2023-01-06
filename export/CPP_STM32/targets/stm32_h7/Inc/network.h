/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/
#ifndef NETWORK_H
#define NETWORK_H

#include "typedefs.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif 


void propagate(const void* inputs, int32_t* outputs);

#ifdef __cplusplus
}
#endif 

#endif
