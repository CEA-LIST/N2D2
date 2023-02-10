/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/


#include "Quantizer/QAT/Activation/QuantizerActivation_Frame.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_QuantizerActivation_Frame(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("QuantizerActivation_Frame_" + typeStr);
    py::class_<QuantizerActivation_Frame<T>, std::shared_ptr<QuantizerActivation_Frame<T>>, QuantizerActivation> (m, pyClassName.c_str(), py::multiple_inheritance()) 
    .doc() = "QuantizerActivation_Frame is the abstract base object for a non-CUDA activation quantizer"
    ;
}

void init_QuantizerActivation_Frame(py::module &m) {
    declare_QuantizerActivation_Frame<float>(m, "float");
#if SIZE_MAX != 0xFFFFFFFF
    declare_QuantizerActivation_Frame<double>(m, "double");
#endif
}

}

