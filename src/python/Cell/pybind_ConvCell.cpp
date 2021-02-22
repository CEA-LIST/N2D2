/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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
#include "Cell/ConvCell.hpp"

#include "Solver/Solver.hpp"
#include "Filler/Filler.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_ConvCell(py::module &m) {
    py::class_<ConvCell, std::shared_ptr<ConvCell>, Cell> convCell (m, "ConvCell", py::multiple_inheritance());

    py::enum_<ConvCell::WeightsExportFormat>(convCell, "WeightsExportFormat")
    .value("OCHW", ConvCell::WeightsExportFormat::OCHW)
    .value("HWCO", ConvCell::WeightsExportFormat::HWCO)
    .export_values();

     convCell
    .def("setWeight", &ConvCell::setWeight, py::arg("output"), py::arg("channel"), py::arg("value"))
    .def("setBias", &ConvCell::setBias, py::arg("output"), py::arg("value"))
    .def("setWeightsSolver", &ConvCell::setWeightsSolver, py::arg("solver"))
    .def("getWeightsSolver", &ConvCell::getWeightsSolver)
    .def("setWeightsFiller", &ConvCell::setWeightsFiller, py::arg("filler"))
    .def("setBiasSolver", &ConvCell::setBiasSolver, py::arg("solver"))
    .def("getBiasSolver", &ConvCell::getBiasSolver)
    .def("setBiasFiller", &ConvCell::setBiasFiller, py::arg("filler"))
    .def("setQuantizer", &ConvCell::setQuantizer, py::arg("quantizer"))
    .def("getQuantizer", &ConvCell::getQuantizer)
    .def("getKernelWidth", &ConvCell::getKernelWidth)
    .def("getKernelHeight", &ConvCell::getKernelHeight)
    .def("getSubSampleX", &ConvCell::getSubSampleX)
    .def("getSubSampleY", &ConvCell::getSubSampleY)
    .def("getStrideX", &ConvCell::getStrideX)
    .def("getStrideY", &ConvCell::getStrideY)
    .def("getPaddingX", &ConvCell::getPaddingX)
    .def("getPaddingY", &ConvCell::getPaddingY)
    .def("getDilationX", &ConvCell::getDilationX)
    .def("getDilationY", &ConvCell::getDilationY)
    //.def("importFreeParameters", &ConvCell::importFreeParameters, py::arg("fileName"), py::arg("ignoreNotExists")=false)
    //.def("exportFreeParameters", &ConvCell::exportFreeParameters, py::arg("fileName"))
    ;

}
}
#endif

