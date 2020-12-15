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
#include "Cell/PaddingCell_Frame.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_PaddingCell_Frame(py::module &m) {
    py::class_<PaddingCell_Frame, std::shared_ptr<PaddingCell_Frame>, PaddingCell,  Cell_Frame<Float_T>> (m,"PaddingCell_Frame", py::multiple_inheritance()) 
    .def(py::init<
    const DeepNet&, 
    const std::string&,
    unsigned int,
    int,
    int,
    int,
    int>(),
         py::arg("deepNet"),
         py::arg("name"),
         py::arg("nbOutputs"),
         py::arg("topPad") = 0,
         py::arg("botPad") = 0,
         py::arg("leftPad") = 0,
         py::arg("rightPad") = 0
         );
}

}
#endif

