/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifdef ONNX

#include <string>
#include "Export/ONNX/ONNX_Config.hpp"


const std::string N2D2::ONNX_Config::IMPLICIT_CASTING = "ImplicitCasting";
const bool N2D2::ONNX_Config::IMPLICIT_CASTING_DEFAULT = true;

const std::string N2D2::ONNX_Config::FAKE_QUANTIZATION = "FakeQuantization";
const bool N2D2::ONNX_Config::FAKE_QUANTIZATION_DEFAULT = false;

#endif
