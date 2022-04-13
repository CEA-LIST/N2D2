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
#include "Cell/DeconvCell.hpp"

#include "Solver/Solver.hpp"
#include "Filler/Filler.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_DeconvCell(py::module &m) {

    py::class_<DeconvCell, std::shared_ptr<DeconvCell>, Cell> (m, "DeconvCell", py::multiple_inheritance())
    .def("getWeight", &DeconvCell::getWeight, py::arg("output"), py::arg("channel"), py::arg("value"))
    .def("setWeight", &DeconvCell::setWeight, py::arg("output"), py::arg("channel"), py::arg("value"))
    .def("setBias", &DeconvCell::setBias, py::arg("output"), py::arg("value"))
    .def("getBias", &DeconvCell::getBias, py::arg("output"), py::arg("value"))
    .def("getBiases", &DeconvCell::getBiases)

    .def("setWeightsSolver", &DeconvCell::setWeightsSolver, py::arg("solver"))
    .def("getWeightsSolver", &DeconvCell::getWeightsSolver)
    .def("setWeightsFiller", &DeconvCell::setWeightsFiller, py::arg("filler"))
    .def("getWeightsFiller", &DeconvCell::getWeightsFiller)
    .def("setBiasSolver", &DeconvCell::setBiasSolver, py::arg("solver"))
    .def("getBiasSolver", &DeconvCell::getBiasSolver)
    .def("setBiasFiller", &DeconvCell::setBiasFiller, py::arg("filler"))
    .def("getBiasFiller", &DeconvCell::getBiasFiller)
    //.def("setQuantizer", &DeconvCell::setQuantizer, py::arg("quantizer"))
    //.def("getQuantizer", &DeconvCell::getQuantizer)
    .def("getKernelWidth", &DeconvCell::getKernelWidth)
    .def("getKernelHeight", &DeconvCell::getKernelHeight)
    //.def("getSubSampleX", &DeconvCell::getSubSampleX)
    //.def("getSubSampleY", &DeconvCell::getSubSampleY)
    .def("getStrideX", &DeconvCell::getStrideX)
    .def("getStrideY", &DeconvCell::getStrideY)
    .def("getPaddingX", &DeconvCell::getPaddingX)
    .def("getPaddingY", &DeconvCell::getPaddingY)
    .def("getDilationX", &DeconvCell::getDilationX)
    .def("getDilationY", &DeconvCell::getDilationY);
}
}
