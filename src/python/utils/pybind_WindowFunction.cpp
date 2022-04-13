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

#include "utils/WindowFunction.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
    
template<class T>
void declare_WindowFunction(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("WindowFunction_" + typeStr);
    py::class_<WindowFunction<T>, std::shared_ptr<WindowFunction<T>>> (m, pyClassName.c_str(), py::multiple_inheritance());

    const std::string pyrect("Rectangular_" + typeStr);
    py::class_<Rectangular<T>, std::shared_ptr<Rectangular<T>>, WindowFunction<T>> (m, pyrect.c_str(), py::multiple_inheritance())
    .def(py::init<>());

    const std::string pyhann("Hann_" + typeStr);
    py::class_<Hann<T>, std::shared_ptr<Hann<T>>, WindowFunction<T>> (m, pyhann.c_str(), py::multiple_inheritance())
    .def(py::init<>());
        
    const std::string pyhamming("Hamming_" + typeStr);
    py::class_<Hamming<T>, std::shared_ptr<Hamming<T>>, WindowFunction<T>> (m, pyhamming.c_str(), py::multiple_inheritance())
    .def(py::init<>());
        
    const std::string pycosine("Cosine_" + typeStr);
    py::class_<Cosine<T>, std::shared_ptr<Cosine<T>>, WindowFunction<T>> (m, pycosine.c_str(), py::multiple_inheritance())
    .def(py::init<>());  
        
    const std::string pygaussian("Gaussian_" + typeStr);
    py::class_<Gaussian<T>, std::shared_ptr<Gaussian<T>>, WindowFunction<T>> (m, pygaussian.c_str(), py::multiple_inheritance())
    .def(py::init<T>(), py::arg("sigma") = 0.4);

        
    const std::string pyblackman("Blackman_" + typeStr);
    py::class_<Blackman<T>, std::shared_ptr<Blackman<T>>, WindowFunction<T>> (m, pyblackman.c_str(), py::multiple_inheritance())
    .def(py::init<T>(), py::arg("alpha") = 0.16);
    
    const std::string pykaiser("Kaiser_" + typeStr);
    py::class_<Kaiser<T>, std::shared_ptr<Kaiser<T>>, WindowFunction<T>> (m, pykaiser.c_str(), py::multiple_inheritance())
    .def(py::init<T>(), py::arg("beta") = 5.0);

}

void init_WindowFunction(py::module &m) {
    declare_WindowFunction<double>(m, "double");
}
}
