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

#include "Filler/ConstantFiller.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace N2D2 {
template<typename T>
void declare_ConstantFiller(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("ConstantFiller_" + typeStr);

    py::class_<ConstantFiller<T>, std::shared_ptr<ConstantFiller<T>>, Filler> (
            m, pyClassName.c_str(), py::multiple_inheritance()
    )

    .def(py::init<T>(), py::arg("value")=0.0)
    .def("getValue", &ConstantFiller<T>::getValue)
    .def("getDataType", [typeStr](ConstantFiller<T>){return typeStr;})
    ;
}

void init_ConstantFiller(py::module &m) {
    declare_ConstantFiller<float>(m, "float");
    declare_ConstantFiller<double>(m, "double");
}
}



