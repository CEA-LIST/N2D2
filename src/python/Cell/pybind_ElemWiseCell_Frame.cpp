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
#include "Cell/ElemWiseCell_Frame.hpp"
#include "Activation/TanhActivation_Frame.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_ElemWiseCell_Frame(py::module &m) {
    py::class_<ElemWiseCell_Frame, std::shared_ptr<ElemWiseCell_Frame>, ElemWiseCell, Cell_Frame<Float_T>> (m, "ElemWiseCell_Frame", py::multiple_inheritance()) 
    .def(py::init<const DeepNet&, 
    const std::string&, 
    unsigned int, 
    ElemWiseCell::Operation,
    const std::vector<Float_T>&,
    const std::vector<Float_T>&,
    const std::shared_ptr<Activation>&>(),
        py::arg("deepNet"), 
        py::arg("name"), 
        py::arg("nbOutputs"), 
        py::arg("operation") = ElemWiseCell::Operation::Sum,
        py::arg("weights") = std::vector<Float_T>(),
        py::arg("shifts") = std::vector<Float_T>(),
        py::arg("activation") = std::shared_ptr<Activation>());
}

}
#endif
