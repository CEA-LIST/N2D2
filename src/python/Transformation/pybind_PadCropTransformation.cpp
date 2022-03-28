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

#include "Transformation/PadCropTransformation.hpp"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace N2D2 {
void init_PadCropTransformation(py::module &m) {
    py::class_<PadCropTransformation, std::shared_ptr<PadCropTransformation>, Transformation> trans (m, "PadCropTransformation", py::multiple_inheritance());

    py::enum_<PadCropTransformation::BorderType>(trans, "BorderType")
    .value("ConstantBorder", PadCropTransformation::BorderType::ConstantBorder)
    .value("ReplicateBorder", PadCropTransformation::BorderType::ReplicateBorder)
    .value("ReflectBorder", PadCropTransformation::BorderType::ReflectBorder)
    .value("WrapBorder", PadCropTransformation::BorderType::WrapBorder)
    .value("MinusOneReflectBorder", PadCropTransformation::BorderType::MinusOneReflectBorder)
    .value("MeanBorder", PadCropTransformation::BorderType::MeanBorder)
    .export_values();

    trans
    .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
    .def("getWidth", &PadCropTransformation::getWidth)
    .def("getHeight", &PadCropTransformation::getHeight)
    .def("getAdditiveWH", &PadCropTransformation::getAdditiveWH)
    .def("getBorderType", &PadCropTransformation::getBorderType)
    .def("getBorderValue", &PadCropTransformation::getBorderValue)
    ;

    // .def("apply", &PadCropTransformation::apply, py::arg("frame"), py::arg("labels"), py::arg("labelsROI"), py::arg("id"));

}
}
