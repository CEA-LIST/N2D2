/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/
#ifndef N2D2_CPP_TENSORRT_CONFIG_H
#define N2D2_CPP_TENSORRT_CONFIG_H

#include <string>

namespace N2D2 {

class CPP_TensorRT_Config {
public:
    static const std::string GEN_STIMULI_CALIB;
    static const bool GEN_STIMULI_CALIB_DEFAULT;

};
}

#endif