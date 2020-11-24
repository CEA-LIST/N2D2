/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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
#include "Cell/FcCell_Frame.hpp"
#include "Activation/TanhActivation_Frame.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_FcCell_Frame(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("FcCell_Frame_" + typeStr);
    py::class_<FcCell_Frame<T>, std::shared_ptr<FcCell_Frame<T>>, FcCell, Cell_Frame<T>> (m, pyClassName.c_str(), py::multiple_inheritance()) 
    .def(py::init<const DeepNet&, const std::string&, unsigned int, const std::shared_ptr<Activation>&>(),
         py::arg("deepNet"), py::arg("name"), py::arg("nbOutputs"), py::arg("activation"));

}

void init_FcCell_Frame(py::module &m) {
    declare_FcCell_Frame<float>(m, "float");
    declare_FcCell_Frame<double>(m, "double");
}
}
#endif

#endif
