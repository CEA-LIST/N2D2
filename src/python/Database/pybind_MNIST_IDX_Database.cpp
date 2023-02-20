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

#include "Database/MNIST_IDX_Database.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

#if SIZE_MAX != 0xFFFFFFFF

namespace N2D2 {
void init_MNIST_IDX_Database(py::module &m) {
    py::class_<MNIST_IDX_Database, std::shared_ptr<MNIST_IDX_Database>, Database>(m, "MNIST_IDX_Database")
        .def(py::init<double>(), py::arg("validation") = 0.0)
        .def(py::init<const std::string&, const std::string&, bool, double>(), py::arg("dataPath"), py::arg("labelPath") = "", py::arg("extractROIs") = false, py::arg("validation") = 0.0);

}
}

#endif  // SIZE_MAX
