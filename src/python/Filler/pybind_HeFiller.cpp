/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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
#include "Filler/HeFiller.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace N2D2 {
template<typename T>
void declare_HeFiller(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("HeFiller_" + typeStr);

    py::class_<HeFiller<T>, std::shared_ptr<HeFiller<T>>, Filler> filler(
            m, pyClassName.c_str(), py::multiple_inheritance()
    );

    py::enum_<class HeFiller<T>::VarianceNorm>(filler, "VarianceNorm")
    .value("FanIn", HeFiller<T>::VarianceNorm::FanIn)
    .value("Average", HeFiller<T>::VarianceNorm::Average)
    .value("FanOut", HeFiller<T>::VarianceNorm::FanOut)
    .export_values();

    filler
    .def(py::init<typename HeFiller<T>::VarianceNorm, T, T>(), 
        py::arg("varianceNorm") = HeFiller<T>::VarianceNorm::FanIn, 
        py::arg("meanNorm") = 0.0, 
        py::arg("scaling") = 1.0)
    .def("getVarianceNorm", &HeFiller<T>::getVarianceNorm)
    .def("getMeanNorm", &HeFiller<T>::getMeanNorm)
    .def("getScaling", &HeFiller<T>::getScaling)
    ;
}

void init_HeFiller(py::module &m) {
    declare_HeFiller<float>(m, "float");
    declare_HeFiller<double>(m, "double");
}
}

#endif
