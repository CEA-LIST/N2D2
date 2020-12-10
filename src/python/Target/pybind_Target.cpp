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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


namespace N2D2 {
void init_Target(py::module &m) {
    py::class_<Target, std::shared_ptr<Target>>(m, "Target", py::multiple_inheritance())
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
    .def("getName", &Target::getName)
    .def("getType", &Target::getType)
    .def("getCell", &Target::getCell)
    .def("getStimuliProvider", &Target::getStimuliProvider)
    .def("getNbTargets", &Target::getNbTargets)
    .def("getTargetTopN", &Target::getTargetTopN)
    .def("getTargetValue", &Target::getTargetValue)
    .def("getDefaultValue", &Target::getDefaultValue)
    .def("setMaskLabelTarget", &Target::setMaskLabelTarget, py::arg("target"))
    .def("labelsMapping", &Target::labelsMapping, py::arg("fileName"), py::arg("createMissingLabels") = false)
    .def("setLabelTarget", &Target::setLabelTarget, py::arg("label"), py::arg("output"))
    .def("setDefaultTarget", &Target::setDefaultTarget, py::arg("output"))
    .def("getLabelTarget", &Target::getLabelTarget, py::arg("label"))
    .def("getDefaultTarget", &Target::getDefaultTarget)
    .def("getTargetLabels", &Target::getTargetLabels, py::arg("output"))
    .def("getTargetLabelsName", &Target::getTargetLabelsName)
    .def("logLabelsMapping", &Target::logLabelsMapping, py::arg("fileName"))
    .def("provideTargets", &Target::provideTargets, py::arg("set"),
     R"mydelimiter(
     Set the output target of a cell. Also create the mapping if labelsMapping option is defined.
     
     :param set: Set of stimuli
     :type set: :py:class:`N2D2.Database.StimuliSet`
    )mydelimiter")
    .def("process", &Target::process, py::arg("set"),
     R"mydelimiter(
     Apply the loss to the cell the target is bind to.
     
     :param set: Set of stimuli
     :type set: :py:class:`N2D2.Database.StimuliSet`
    )mydelimiter")
    .def("logEstimatedLabels", &Target::logEstimatedLabels, py::arg("dirName"))
    .def("logEstimatedLabelsJSON", &Target::logEstimatedLabelsJSON, py::arg("dirName"), py::arg("fileName") = "", py::arg("xOffset") = 0, py::arg("yOffset") = 0, py::arg("append") = false)
    .def("logLabelsLegend", &Target::logLabelsLegend, py::arg("fileName"))
    .def("getEstimatedLabels", (const Target::TensorLabels_T& (Target::*)() const) &Target::getEstimatedLabels)
    .def("getEstimatedLabelsValue", &Target::getEstimatedLabelsValue)
    .def("getEstimatedLabels", (Target::TensorLabelsValue_T (Target::*)(const std::shared_ptr<ROI>&, unsigned int, Float_T*) const) &Target::getEstimatedLabels, py::arg("roi"), py::arg("batchPos") = 0, py::arg("values") = NULL)
    .def("getEstimatedLabel", &Target::getEstimatedLabel, py::arg("roi"), py::arg("batchPos") = 0, py::arg("values") = NULL)
    .def("getLoss", &Target::getLoss)
    .def("log", &Target::log, py::arg("fileName"), py::arg("set"))
    .def("clear", &Target::clear, py::arg("set"));
}
}
#endif
