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
#include "Transformation/RangeAffineTransformation.hpp"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_RangeAffineTransformation(py::module &m) {
    py::class_<RangeAffineTransformation, std::shared_ptr<RangeAffineTransformation>, Transformation> rat (m, "RangeAffineTransformation", py::multiple_inheritance());
   
    py::enum_<RangeAffineTransformation::Operator>(rat, "Operator")
    .value("Plus", RangeAffineTransformation::Operator::Plus)
    .value("Minus", RangeAffineTransformation::Operator::Minus)
    .value("Multiplies", RangeAffineTransformation::Operator::Multiplies)
    .value("Divides", RangeAffineTransformation::Operator::Divides)
    .export_values();

    rat
    .def(py::init<RangeAffineTransformation::Operator, const std::vector<double>&, RangeAffineTransformation::Operator, const std::vector<double>&>(), py::arg("firstOperator"), py::arg("firstValue"), py::arg("secondOperator") = RangeAffineTransformation::Operator::Plus, py::arg("secondValue") = std::vector<double>())
    .def(py::init<RangeAffineTransformation::Operator, double, RangeAffineTransformation::Operator, double>(), py::arg("firstOperator"), py::arg("firstValue"), py::arg("secondOperator") = RangeAffineTransformation::Operator::Plus, py::arg("secondValue") = 0.0)
    .def(py::init<const RangeAffineTransformation&>(), py::arg("trans"));
}
}
#endif
