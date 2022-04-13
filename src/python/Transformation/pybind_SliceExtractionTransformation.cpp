/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Cyril MOINEAU (cyril.moineau@cea.fr)

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
#include "Transformation/SliceExtractionTransformation.hpp"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_SliceExtractionTransformation(py::module &m) {
    py::class_<SliceExtractionTransformation, std::shared_ptr<SliceExtractionTransformation>, Transformation> set (m, "SliceExtractionTransformation", py::multiple_inheritance());
    
    
    py::enum_<SliceExtractionTransformation::BorderType>(set, "BorderType")
    .value("ConstantBorder", SliceExtractionTransformation::BorderType::ConstantBorder)
    .value("ReplicateBorder", SliceExtractionTransformation::BorderType::ReplicateBorder)
    .value("ReflectBorder", SliceExtractionTransformation::BorderType::ReflectBorder)
    .value("WrapBorder", SliceExtractionTransformation::BorderType::WrapBorder)
    .value("MinusOneReflectBorder", SliceExtractionTransformation::BorderType::MinusOneReflectBorder)
    .value("MeanBorder", SliceExtractionTransformation::BorderType::MeanBorder)
    .export_values();

    set.def(py::init<unsigned int, unsigned int, unsigned int, unsigned int>(), py::arg("width"), py::arg("height"), py::arg("OffsetX") = 0,  py::arg("OffsetY") = 0)
    .def(py::init<SliceExtractionTransformation&>(), py::arg("trans"))
    .def("getWidth", &SliceExtractionTransformation::getWidth)
    .def("getHeight", &SliceExtractionTransformation::getHeight)
    .def("getOffsetX", &SliceExtractionTransformation::getOffsetX)
    .def("getOffsetY", &SliceExtractionTransformation::getOffsetY)
    .def("getRandomOffsetX", &SliceExtractionTransformation::getRandomOffsetX)
    .def("getRandomOffsetY", &SliceExtractionTransformation::getRandomOffsetY)
    .def("getRandomRotation", &SliceExtractionTransformation::getRandomRotation)
    .def("getRandomRotationRange", &SliceExtractionTransformation::getRandomRotationRange)
    .def("getRandomScaling", &SliceExtractionTransformation::getRandomScaling)
    .def("getRandomScalingRange", &SliceExtractionTransformation::getRandomScalingRange)
    .def("getAllowPadding", &SliceExtractionTransformation::getAllowPadding)
    .def("getBorderType", &SliceExtractionTransformation::getBorderType)
    .def("getBorderValue", &SliceExtractionTransformation::getBorderValue)
    ;
}
}
