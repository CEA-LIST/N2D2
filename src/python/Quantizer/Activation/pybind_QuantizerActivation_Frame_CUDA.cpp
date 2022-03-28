/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifdef CUDA

#include "Quantizer/Activation/QuantizerActivation_Frame_CUDA.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_QuantizerActivation_Frame_CUDA(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("QuantizerActivation_Frame_CUDA_" + typeStr);
    py::class_<QuantizerActivation_Frame_CUDA<T>, std::shared_ptr<QuantizerActivation_Frame_CUDA<T>>, QuantizerActivation> (m, pyClassName.c_str(), py::multiple_inheritance()) 
    .doc() = "QuantizerActivation_Frame_CUDA is the abstract base object for a CUDA activation quantizer"
    ;
}

void init_QuantizerActivation_Frame_CUDA(py::module &m) {
    declare_QuantizerActivation_Frame_CUDA<float>(m, "float");
    declare_QuantizerActivation_Frame_CUDA<double>(m, "double");
}

}


#endif
