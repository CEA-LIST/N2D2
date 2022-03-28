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

#include "Cell/ElemWiseCell.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace N2D2 {
void init_ElemWiseCell(py::module &m) {

    py::class_<ElemWiseCell, std::shared_ptr<ElemWiseCell>, Cell> ewc (m, "ElemWiseCell", py::multiple_inheritance());
    
    py::enum_<ElemWiseCell::Operation>(ewc, "Operation")
    .value("Sum", ElemWiseCell::Operation::Sum)
    .value("AbsSum", ElemWiseCell::Operation::AbsSum)
    .value("EuclideanSum", ElemWiseCell::EuclideanSum)
    .value("Prod", ElemWiseCell::Operation::Prod)
    .value("Max", ElemWiseCell::Operation::Max)
    .export_values();

    py::enum_<ElemWiseCell::CoeffMode>(ewc, "CoeffMode")
    .value("PerLayer", ElemWiseCell::CoeffMode::PerLayer)
    .value("PerInput", ElemWiseCell::CoeffMode::PerInput)
    .value("PerChannel", ElemWiseCell::CoeffMode::PerChannel)
    .export_values();

    ewc
    .def("getOperation", &ElemWiseCell::getOperation)
    .def("getCoeffMode", &ElemWiseCell::getCoeffMode)
    .def("getWeights", &ElemWiseCell::getWeights)
    .def("getShifts", &ElemWiseCell::getShifts);
}
}

