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


#include "Quantizer/QAT/Activation/SAT/SATQuantizerActivation.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace N2D2 {
void init_SATQuantizerActivation(py::module &m) {
    py::class_<
        SATQuantizerActivation, 
        std::shared_ptr<SATQuantizerActivation>> (m, "SATQuantizerActivation", py::multiple_inheritance())
    .def("setSolver", &SATQuantizerActivation::setSolver, py::arg("solver"))
    .def("getSolver", &SATQuantizerActivation::getSolver)
    .def("getAlphaParameter", &SATQuantizerActivation::getAlphaParameter)
    .def("exportParameters", &SATQuantizerActivation::exportParameters, py::arg("dirName"), py::arg("cellName"))
    .def("importParameters", &SATQuantizerActivation::importParameters, py::arg("dirName"), py::arg("cellName"), py::arg("ignoreNotExists"))
    ;
;
}

}


