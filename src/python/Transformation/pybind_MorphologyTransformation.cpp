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

#ifdef PYBIND
#include "Transformation/MorphologyTransformation.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_MorphologyTransformation(py::module &m) {
    py::class_<MorphologyTransformation, std::shared_ptr<MorphologyTransformation>, Transformation> mt (m, "MorphologyTransformation", py::multiple_inheritance());

    py::enum_<MorphologyTransformation::Operation>(mt, "Operation")
    .value("Erode", MorphologyTransformation::Operation::Erode)
    .value("Dilate", MorphologyTransformation::Operation::Dilate)
    .value("Opening", MorphologyTransformation::Operation::Opening)
    .value("Closing", MorphologyTransformation::Operation::Closing)
    .value("Gradient", MorphologyTransformation::Operation::Gradient)
    .value("TopHat", MorphologyTransformation::Operation::TopHat)
    .value("BlackHat", MorphologyTransformation::Operation::BlackHat)
    .export_values();

    mt
    .def(py::init<MorphologyTransformation::Operation, unsigned int, bool>(), py::arg("operation"), py::arg("size"), py::arg("applyToLabels") = false)
    .def(py::init<const MorphologyTransformation&>(), py::arg("trans"));
}
}
#endif

