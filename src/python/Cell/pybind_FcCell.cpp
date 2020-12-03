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

#ifdef CUDA

#ifdef PYBIND
#include "Cell/FcCell.hpp"

// These includes are necessary for the bindings using abstract classes
#include "Solver/Solver.hpp"
#include "Filler/Filler.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {

void init_FcCell(py::module &m) {
    py::class_<FcCell, std::shared_ptr<FcCell>, Cell> (m, "FcCell", py::multiple_inheritance())
        .def("setWeightsSolver", &FcCell::setWeightsSolver, py::arg("solver"))
        .def("getWeightsSolver", &FcCell::getWeightsSolver)
        .def("setWeightsFiller", &FcCell::setWeightsFiller, py::arg("filler"))
        .def("getWeightsFiller", &FcCell::setWeightsFiller)
        .def("setBiasSolver", &FcCell::setBiasSolver, py::arg("solver"))
        .def("getBiasSolver", &FcCell::getBiasSolver)
        .def("setBiasFiller", &FcCell::setBiasFiller, py::arg("filler"))
        .def("getBiasFiller", &FcCell::setBiasFiller);

}
}
#endif

#endif



 