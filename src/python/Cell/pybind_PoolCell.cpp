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

#include "Cell/PoolCell.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_PoolCell(py::module &m) {

    py::class_<PoolCell, std::shared_ptr<PoolCell>, Cell> pool (m, "PoolCell", py::multiple_inheritance());
    pool
    .def("getPoolHeight", &PoolCell::getPoolHeight)
    .def("getPoolWidth", &PoolCell::getPoolWidth)
    .def("getStrideX", &PoolCell::getStrideX)
    .def("getStrideY", &PoolCell::getStrideY)
    .def("getPaddingX", &PoolCell::getPaddingX)
    .def("getPaddingY", &PoolCell::getPaddingY)
    .def("getPooling", &PoolCell::getPooling)
    ;
    py::enum_<PoolCell::Pooling>(pool, "Pooling")
    .value("Max", PoolCell::Pooling::Max)
    .value("Average", PoolCell::Pooling::Average)
    .export_values();

}
}
