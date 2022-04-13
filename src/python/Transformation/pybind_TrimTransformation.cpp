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
#include "Transformation/TrimTransformation.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_TrimTransformation(py::module &m) {
    // TODO : Need to bind OpenCV, can't even create a TrimTransformation with default arg because it's not yet convertible to python.
    py::class_<TrimTransformation, std::shared_ptr<TrimTransformation>, Transformation> (m, "TrimTransformation", py::multiple_inheritance())
    .def(py::init<unsigned int, const cv::Mat&>(), py::arg("nbLevels"), py::arg("kernel") = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)))
    .def(py::init<TrimTransformation&>(), py::arg("trans"));
}
}
