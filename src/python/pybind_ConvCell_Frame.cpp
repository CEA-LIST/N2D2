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
#include "Cell/ConvCell_Frame.hpp"
#include "Activation/TanhActivation_Frame.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_ConvCell_Frame(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("ConvCell_Frame_" + typeStr);
    py::class_<ConvCell_Frame<T>, std::shared_ptr<ConvCell_Frame<T>>, ConvCell, Cell_Frame<T>> (m, pyClassName.c_str(), py::multiple_inheritance()) 
    // TODO : Declare std::make_shared<TanhActivation_Frame<Float_T> >() as a default argument for activation if not remove the include TanhActivation
    .def(py::init<const DeepNet&, const std::string&, const std::vector<unsigned int>&, unsigned int, const std::vector<unsigned int>&, const std::vector<unsigned int>&, const std::vector<int>&, const std::vector<unsigned int>&, const std::shared_ptr<Activation>&>(),
         py::arg("deepNet"), py::arg("name"), py::arg("kernelDims"), py::arg("nbOutputs"), 
         py::arg("subSampleDims") = std::vector<unsigned int>(2, 1U), py::arg("strideDims") = std::vector<unsigned int>(2, 1U), 
         py::arg("paddingDims") = std::vector<int>(2, 0), py::arg("dilationDims") = std::vector<unsigned int>(2, 1U),
         py::arg("activation"))
    .def("propagate", &ConvCell_Frame<T>::propagate, py::arg("inference") = false)
    .def("backPropagate", &ConvCell_Frame<T>::backPropagate)
    .def("update", &ConvCell_Frame<T>::update)
    .def("initialize", &ConvCell_Frame<T>::initialize);

}

void init_ConvCell_Frame(py::module &m) {
    declare_ConvCell_Frame<float>(m, "float");
    declare_ConvCell_Frame<double>(m, "double");
}
}
#endif

#endif
