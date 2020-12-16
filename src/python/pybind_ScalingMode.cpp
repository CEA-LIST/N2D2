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
#include "Scaling.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_ScalingMode(py::module &m) {
    py::enum_<ScalingMode>(m, "ScalingMode", py::arithmetic())
    .value("NONE", ScalingMode::NONE)
    .value("FLOAT_MULT", ScalingMode::FLOAT_MULT)
    .value("FIXED_MULT", ScalingMode::FIXED_MULT)
    .value("SINGLE_SHIFT", ScalingMode::SINGLE_SHIFT)
    .value("DOUBLE_SHIFT", ScalingMode::DOUBLE_SHIFT);
}
}
#endif

#endif
