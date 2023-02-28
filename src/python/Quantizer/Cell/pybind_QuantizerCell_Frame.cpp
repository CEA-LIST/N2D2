/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/


#include "Quantizer/QAT/Cell/QuantizerCell_Frame.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_QuantizerCell_Frame(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("QuantizerCell_Frame_" + typeStr);
    py::class_<QuantizerCell_Frame<T>, std::shared_ptr<QuantizerCell_Frame<T>>, QuantizerCell> (m, pyClassName.c_str(), py::multiple_inheritance()) 
    .doc() = "QuantizerCell_Frame is the abstract base object for a non-CUDA cell quantizer"
    ;
}

void init_QuantizerCell_Frame(py::module &m) {
    declare_QuantizerCell_Frame<float>(m, "float");
#if SIZE_MAX != 0xFFFFFFFF
    declare_QuantizerCell_Frame<double>(m, "double");
#endif
}

}

