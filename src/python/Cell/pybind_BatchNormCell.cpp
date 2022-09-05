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

#include "Cell/BatchNormCell.hpp"

#include "Solver/Solver.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_BatchNormCell(py::module &m) {
    py::class_<BatchNormCell, std::shared_ptr<BatchNormCell>, Cell> (m, "BatchNormCell", py::multiple_inheritance())
    .def("setScaleSolver", &BatchNormCell::setScaleSolver, py::arg("solver"))
    .def("getScaleSolver", &BatchNormCell::getScaleSolver)
    .def("setBiasSolver", &BatchNormCell::setBiasSolver, py::arg("solver"))
    .def("getBiasSolver", &BatchNormCell::getBiasSolver)
    .def("getBiases", &BatchNormCell::getBiases)
    .def("getScales", &BatchNormCell::getScales)
    .def("getMeans", &BatchNormCell::getMeans)
    .def("getVariances", &BatchNormCell::getVariances)
    .def("setVariance", &BatchNormCell::setVariance, py::arg("index"), py::arg("variance"))
    .def("setMean", &BatchNormCell::setMean, py::arg("index"), py::arg("mean"))
    .def("setScale", &BatchNormCell::setScale, py::arg("index"), py::arg("scale"))
    .def("setBias", &BatchNormCell::setBias, py::arg("index"), py::arg("bias"))
    .def("getMovingAverageMomentum", &BatchNormCell::getMovingAverageMomentum)
    .def("getEpsilon", &BatchNormCell::getEpsilon);
}
}
