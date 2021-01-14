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
#include "utils/Parameterizable.hpp"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_Parameterizable(py::module &m) {

    // TODO : Parameter_T binding maybe deprecated !
    py::class_<Parameter_T, std::shared_ptr<Parameter_T>> (m, "Parameter_T", py::multiple_inheritance())
    // .def_readwrite("mValue", &Parameter_T::mValue) // => Doesn't give values (Capsule NULL)
    .def("getType",
    [](Parameter_T& param){
        const char* type = param.mType->name();
        return type;
    })
    ;

    py::class_<Parameterizable, std::shared_ptr<Parameterizable>> (m, "Parameterizable", py::multiple_inheritance())
    .def("setParameter", (void (Parameterizable::*)(const std::string&, const std::string&)) (&Parameterizable::setParameter), py::arg("name"), py::arg("value"))
    // .def("getParameter", &Parameterizable::getParameter<std::string>, py::arg("name")) //Compiling but error during excetution
    .def("getParameterAndType", &Parameterizable::getParameterAndType, py::arg("name")) 
    .def("getParameters", &Parameterizable::getParameters)
    .def_readwrite("mParameters", &Parameterizable::mParameters)
    // .def("getTypeParameters",
    // [](const Parameterizable& p, const std::string& name){
    //     return py::type::of<p.getParameter(name)>();
    // })
    ;
    
}
}
#endif

#endif
