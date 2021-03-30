/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Cyril MOINEAU (cyril.moineau@cea.fr)
                    Victor GACOIN

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
#include "Cell/Cell_Frame_CUDA.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "StimuliProvider.hpp"

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_Cell_Frame_CUDA(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("Cell_Frame_CUDA_" + typeStr);
    py::class_<Cell_Frame_CUDA<T>, std::shared_ptr<Cell_Frame_CUDA<T>>, Cell, Cell_Frame_Top> (m, pyClassName.c_str(), py::multiple_inheritance())
    .def("setDiffInputs", &Cell_Frame_CUDA<T>::setDiffInputs, py::arg("diffInput"))
    .def("setDiffInputsValid", &Cell_Frame_CUDA<T>::setDiffInputsValid)
    //.def("addInput", (void (Cell_Frame_CUDA<T>::*)(BaseTensor&, BaseTensor&)) &Cell_Frame_CUDA<T>::addInput, py::arg("inputs"), py::arg("diffOutputs"));
    .def("applyLoss", (double (Cell_Frame_CUDA<T>::*)(double, double)) &Cell_Frame_CUDA<T>::applyLoss, py::arg("targetVal"), py::arg("defaultVal"))
    .def("setOutputTarget",  &Cell_Frame_CUDA<T>::setOutputTarget, py::arg("targets"))

    .def("clearInputTensors", &Cell_Frame_CUDA<T>::clearInputTensors)
    .def("linkInput",  (void (Cell_Frame_CUDA<T>::*)(Cell*)) &Cell_Frame_CUDA<T>::linkInput, py::arg("cell"))
    .def("linkInput",  (void (Cell_Frame_CUDA<T>::*)(StimuliProvider&, unsigned int, unsigned int, unsigned int, unsigned int)) &Cell_Frame_CUDA<T>::linkInput, 
        py::arg("sp"), py::arg("x0")=0, py::arg("y0")=0, py::arg("width")=0, py::arg("height")=0)

    ;


}

    //.def("getOutputs", (BaseTensor& (Cell_Frame_Top::*)()) &Cell_Frame_Top::getOutputs, py::return_value_policy::reference)
    //virtual void addInput(BaseTensor& inputs,
    //                      BaseTensor& diffOutputs);

void init_Cell_Frame_CUDA(py::module &m) {
    declare_Cell_Frame_CUDA<float>(m, "float");
    declare_Cell_Frame_CUDA<double>(m, "double");
}
}
#endif

#endif
