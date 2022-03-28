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

#include "Cell/NormalizeCell_Frame.hpp"
#include "DeepNet.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_NormalizeCell_Frame(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("NormalizeCell_Frame_" + typeStr);
    py::class_<NormalizeCell_Frame<T>, std::shared_ptr<NormalizeCell_Frame<T>>, NormalizeCell, Cell_Frame<T>> (m, pyClassName.c_str(), py::multiple_inheritance()) 
    .def(py::init<const DeepNet&, const std::string&, unsigned int, NormalizeCell::Norm>(),
         py::arg("deepNet"), py::arg("name"), py::arg("nbOutputs"), py::arg("norm"));

}

void init_NormalizeCell_Frame(py::module &m) {
    declare_NormalizeCell_Frame<double>(m, "double"); 
    declare_NormalizeCell_Frame<float>(m, "float"); 

}
}
