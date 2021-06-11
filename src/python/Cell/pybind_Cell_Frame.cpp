/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".
s
    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
*/

#ifdef PYBIND
#include "Cell/Cell_Frame.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "StimuliProvider.hpp"

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_Cell_Frame(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("Cell_Frame_" + typeStr);
    py::class_<Cell_Frame<T>, std::shared_ptr<Cell_Frame<T>>, Cell, Cell_Frame_Top> (m, pyClassName.c_str(), py::multiple_inheritance())
    // .def("getDiffOutputs", &Cell_Frame<T>::getDiffOutputs)
    .def("setDiffInputs", &Cell_Frame<T>::setDiffInputs, py::arg("diffInput"))
    .def("setDiffInputsValid", &Cell_Frame<T>::setDiffInputsValid)

    .def("clearInputTensors", &Cell_Frame<T>::clearInputTensors)
    .def("clearOutputTensors", &Cell_Frame<T>::clearOutputTensors)
    .def("initializeParameters", &Cell_Frame<T>::initializeParameters, py::arg("nbInputChannels"), py::arg("nbInputs"))
    .def("linkInput",  (void (Cell_Frame<T>::*)(Cell*)) &Cell_Frame<T>::linkInput, py::arg("cell"))
    .def("linkInput",  (void (Cell_Frame<T>::*)(StimuliProvider&, unsigned int, unsigned int, unsigned int, unsigned int)) &Cell_Frame<T>::linkInput, 
        py::arg("sp"), py::arg("x0")=0, py::arg("y0")=0, py::arg("width")=0, py::arg("height")=0)
    ;

}
 
void init_Cell_Frame(py::module &m) {
    declare_Cell_Frame<float>(m, "float");
    declare_Cell_Frame<double>(m, "double");
}
}
#endif
