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
#include "Cell/TransformationCell_Frame_CUDA.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_TransformationCell_Frame_CUDA(py::module &m) {
    py::class_<TransformationCell_Frame_CUDA, std::shared_ptr<TransformationCell_Frame_CUDA>, TransformationCell,  Cell_Frame_CUDA<Float_T>> (m, "TransformationCell_Frame_CUDA", py::multiple_inheritance()) 
    .def(py::init<
    const DeepNet&, 
    const std::string&,
    unsigned int,
    const std::shared_ptr<Transformation>&
    >(),
    py::arg("deepNet"),
    py::arg("name"),
    py::arg("nbOutputs"),
    py::arg("transformation")
    );
}
}
#endif

#endif

