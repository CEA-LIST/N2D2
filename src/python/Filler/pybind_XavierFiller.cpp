/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Filler/XavierFiller.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace N2D2 {
template<typename T>
void declare_XavierFiller(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("XavierFiller_" + typeStr);

    py::class_<XavierFiller<T>, std::shared_ptr<XavierFiller<T>>, Filler> filler(
            m, pyClassName.c_str(), py::multiple_inheritance()
    );

    py::enum_<enum XavierFiller<T>::VarianceNorm>(filler, "VarianceNorm")
    .value("FanIn", XavierFiller<T>::VarianceNorm::FanIn)
    .value("Average", XavierFiller<T>::VarianceNorm::Average)
    .value("FanOut", XavierFiller<T>::VarianceNorm::FanOut)
    .export_values();

    py::enum_<enum XavierFiller<T>::Distribution>(filler, "Distribution")
    .value("Uniform", XavierFiller<T>::Distribution::Uniform)
    .value("Normal", XavierFiller<T>::Distribution::Normal)
    .export_values();

    
    filler
    .def(py::init<typename XavierFiller<T>::VarianceNorm, typename XavierFiller<T>::Distribution, T>(), 
        py::arg("varianceNorm") = XavierFiller<T>::VarianceNorm::FanIn, 
        py::arg("distribution") = XavierFiller<T>::Distribution::Uniform, 
        py::arg("scaling") = 1.0) 
    .def("getVarianceNorm", &XavierFiller<T>::getVarianceNorm)
    .def("getDistribution", &XavierFiller<T>::getDistribution)
    .def("getScaling", &XavierFiller<T>::getScaling)
    .def("getDataType", [typeStr](XavierFiller<T>){return typeStr;})
    ;
}

void init_XavierFiller(py::module &m) {
    declare_XavierFiller<float>(m, "float");
#if SIZE_MAX != 0xFFFFFFFF
    declare_XavierFiller<double>(m, "double");
#endif
}
}



