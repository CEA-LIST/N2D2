/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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

#ifdef PYBIND
#include "Activation/RectifierActivation_Frame_CUDA.hpp"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
template<class T>
void declare_RectifierActivation_Frame_CUDA(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("RectifierActivation_Frame_CUDA_" + typeStr);

    py::class_<RectifierActivation_Frame_CUDA<T>, std::shared_ptr<RectifierActivation_Frame_CUDA<T>>, RectifierActivation> (m, pyClassName.c_str(), py::multiple_inheritance())
    .def(py::init<>());
    //.def("create", &RectifierActivation_Frame_CUDA<T>::create)
    //.def("propagate", &RectifierActivation_Frame_CUDA<T>::propagate, py::arg("cell"), py::arg("data"), py::arg("inference") = false)
    //.def("backPropagate", &RectifierActivation_Frame_CUDA<T>::backPropagate, py::arg("cell"), py::arg("data"), py::arg("diffData"));
}

void init_RectifierActivation_Frame_CUDA(py::module &m) {
    declare_RectifierActivation_Frame_CUDA<float>(m, "float");
    declare_RectifierActivation_Frame_CUDA<double>(m, "double");
}

}
#endif

#endif