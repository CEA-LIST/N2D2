/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
*/

#ifdef CUDA

#include "Quantizer/QAT/Cell/LSQ/LSQQuantizerCell_Frame_CUDA.hpp"
#include "Quantizer/QAT/Cell/LSQ/LSQQuantizerCell.hpp"
#include "Quantizer/QAT/Cell/QuantizerCell_Frame_CUDA.hpp"
#include "Quantizer/QAT/Cell/QuantizerCell.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace N2D2 {
template<typename T>
void declare_LSQQuantizerCell_Frame_CUDA(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("LSQQuantizerCell_Frame_CUDA_" + typeStr);
    py::class_<
        LSQQuantizerCell_Frame_CUDA<T>, 
        std::shared_ptr<LSQQuantizerCell_Frame_CUDA<T>>, 
        LSQQuantizerCell,
        QuantizerCell_Frame_CUDA<T>> (m, pyClassName.c_str(), py::multiple_inheritance()) 
    .def(py::init())
    .def("setSolver", &LSQQuantizerCell_Frame_CUDA<T>::setSolver, py::arg("solver"))
    .def("getSolver", &LSQQuantizerCell_Frame_CUDA<T>::getSolver);
}

void init_LSQQuantizerCell_Frame_CUDA(py::module &m) {
    declare_LSQQuantizerCell_Frame_CUDA<float>(m, "float");
}
}

#endif
