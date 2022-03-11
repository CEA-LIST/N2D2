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
#include "Scaling.hpp"
#include "ScalingMode.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_Scaling(py::module &m) {
    py::class_<AbstractScaling> (m, "AbstractScaling");

    py::class_<FloatingPointScaling, AbstractScaling> (m, "FloatingPointScaling", py::multiple_inheritance())
    .def(py::init<std::vector<Float_T>, bool, std::vector<Float_T>>(), 
        py::arg("scalingPerOutput"), py::arg("isclipped"), py::arg("clipping"))
    .def("getScalingPerOutput", &FloatingPointScaling::getScalingPerOutput)
    .def("getClippingPerOutput", &FloatingPointScaling::getClippingPerOutput)
    .def("getIsClipped", &FloatingPointScaling::getIsClipped);
    
    py::class_<FixedPointScaling, AbstractScaling> (m, "FixedPointScaling", py::multiple_inheritance())
    .def(py::init<std::size_t, std::vector<std::int32_t>, bool, std::vector<Float_T>>(),
        py::arg("nbFractionalBits"), py::arg("scaling"), py::arg("isclipped"), py::arg("clipping"))
    .def("getScalingPerOutput", &FixedPointScaling::getScalingPerOutput)
    .def("getClippingPerOutput", &FixedPointScaling::getClippingPerOutput)
    .def("getIsClipped", &FixedPointScaling::getIsClipped)
    .def("getFractionalBits", &FixedPointScaling::getFractionalBits);
    
    py::class_<SingleShiftScaling, AbstractScaling> (m, "SingleShiftScaling", py::multiple_inheritance())
    .def(py::init<std::vector<unsigned char>, bool, std::vector<Float_T>>(),
        py::arg("scaling"), py::arg("isclipped"), py::arg("clipping"))
    .def("getScalingPerOutput", &SingleShiftScaling::getScalingPerOutput)
    .def("getClippingPerOutput", &SingleShiftScaling::getClippingPerOutput)
    .def("getIsClipped", &SingleShiftScaling::getIsClipped);

    py::class_<DoubleShiftScaling, AbstractScaling> (m, "DoubleShiftScaling", py::multiple_inheritance())
    .def(py::init<std::vector<std::pair<unsigned char, unsigned char>>, bool, std::vector<std::pair<unsigned char, unsigned char>>>(), 
        py::arg("scaling"), py::arg("isclipped"), py::arg("clipping"))
    .def("getScalingPerOutput", &DoubleShiftScaling::getScalingPerOutput)
    .def("getClippingPerOutput", &DoubleShiftScaling::getClippingPerOutput)
    .def("getIsClipped", &DoubleShiftScaling::getIsClipped);

    py::class_<Scaling, std::shared_ptr<Scaling>> (m, "Scaling", py::multiple_inheritance())
    .def(py::init<>());
}
}
#endif

