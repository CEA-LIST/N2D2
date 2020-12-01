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

#ifdef PYBIND
#include "Cell/Cell.hpp"
#include "StimuliProvider.hpp"
#include "Target/Target.hpp"
#include "Target/TargetScore.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


namespace N2D2 {
void init_TargetScore(py::module &m) {
    py::class_<TargetScore, std::shared_ptr<TargetScore>, Target>(m, "TargetScore", py::multiple_inheritance())
    .def(py::init<
        const std::string&, 
        const std::shared_ptr<Cell>&, 
        const std::shared_ptr<StimuliProvider>&,
        double,
        double,
        unsigned int,
        const std::string&,
        bool>(),
        py::arg("name"), 
        py::arg("cell"), 
        py::arg("sp"), 
        py::arg("targetValue") = 1.0,
        py::arg("defaultValue") = 0.0, 
        py::arg("targetTopN") = 1,
        py::arg("labelsMapping") = "", 
        py::arg("createMissingLabels") = false)
    .def("getAverageSuccess", &TargetScore::getAverageSuccess, py::arg("set"), py::arg("avgWindow"))
    .def("process", &TargetScore::process, py::arg("set"));
}
}
#endif

