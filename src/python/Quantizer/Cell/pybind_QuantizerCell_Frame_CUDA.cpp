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

#include "Quantizer/QAT/Cell/QuantizerCell_Frame_CUDA.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_QuantizerCell_Frame_CUDA(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("QuantizerCell_Frame_CUDA_" + typeStr);
    py::class_<QuantizerCell_Frame_CUDA<T>, std::shared_ptr<QuantizerCell_Frame_CUDA<T>>, QuantizerCell> (m, pyClassName.c_str(), py::multiple_inheritance()) 
    .doc() = "QuantizerCell_Frame_CUDA is the abstract base object for a CUDA cell quantizer"
    ;
}

void init_QuantizerCell_Frame_CUDA(py::module &m) {
    declare_QuantizerCell_Frame_CUDA<float>(m, "float");
    declare_QuantizerCell_Frame_CUDA<double>(m, "double");
}

}


#endif




