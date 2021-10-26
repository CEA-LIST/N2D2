/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/
#ifndef N2D2_ONNX_CONFIG_H
#define N2D2_ONNX_CONFIG_H

#ifdef ONNX

#include <string>

namespace N2D2 {

class ONNX_Config {
public:
    static const std::string IMPLICIT_CASTING;
    static const bool IMPLICIT_CASTING_DEFAULT;

    static const std::string FAKE_QUANTIZATION;
    static const bool FAKE_QUANTIZATION_DEFAULT;
};
}

#endif

#endif
