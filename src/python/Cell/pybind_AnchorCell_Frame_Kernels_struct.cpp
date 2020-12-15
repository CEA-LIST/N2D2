// /*
//     (C) Copyright 2020 CEA LIST. All Rights Reserved.
//     Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
//                     Cyril MOINEAU (cyril.moineau@cea.fr)

//     This software is governed by the CeCILL-C license under French law and
//     abiding by the rules of distribution of free software.  You can  use,
//     modify and/ or redistribute the software under the terms of the CeCILL-C
//     license as circulated by CEA, CNRS and INRIA at the following URL
//     "http://www.cecill.info".

//     As a counterpart to the access to the source code and  rights to copy,
//     modify and redistribute granted by the license, users are provided only
//     with a limited warranty  and the software"s author,  the holder of the
//     economic rights,  and the successive licensors  have only  limited
//     liability.

//     The fact that you are presently reading this means that you have had
//     knowledge of the CeCILL-C license and that you accept its terms.
// */


// #ifdef PYBIND
// #include "Cell/AnchorCell_Frame_Kernels_struct.hpp"


// #include <pybind11/pybind11.h>

// namespace py = pybind11;

// namespace N2D2 {
// namespace AnchorCell_Frame_Kernels {
// void init_AnchorCell_Frame_Kernels_struct(py::module &m) {

//     py::class_<Anchor> anchor (m, "Anchor");

//     py::enum_<Anchor::Anchoring>(anchor, "Operator")
//     .value("TopLeft", Anchor::Anchoring::TopLeft)
//     .value("Centered", Anchor::Anchoring::Centered)
//     .value("Original", Anchor::Anchoring::Original)
//     .value("OriginalFlipped", Anchor::Anchoring::OriginalFlipped)
//     .export_values();

//     anchor
//     .def(py::init<>())
//     .def(py::init<float, float, float, float>(), py::arg("x0"), py::arg("y0"), py::arg("width"), py::arg("height"))
//     .def(py::init<float, float, Anchor::Anchoring>(), py::arg("width"), py::arg("height"), py::arg("anchoring") = Anchor::Anchoring::TopLeft)
//     .def(py::init<unsigned int, double, double, Anchor::Anchoring>(), py::arg("area"), py::arg("ratio"), py::arg("scale") = 1.0, py::arg("anchoring") = Anchor::Anchoring::TopLeft)
//     ;
// }
// }
// }
// #endif
 
 
 
