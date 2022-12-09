/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#include "Quantizer/QAT/Cell/Prune/PruneQuantizerCell.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace N2D2 {
void init_PruneQuantizerCell(py::module &m) {
    py::class_<
        PruneQuantizerCell, 
        std::shared_ptr<PruneQuantizerCell>> q(m, "PruneQuantizerCell", py::multiple_inheritance());
    q.def("setPruningMode", &PruneQuantizerCell::setPruningMode, py::arg("prune_mode"));
    q.def("setThreshold", &PruneQuantizerCell::setThreshold, py::arg("threshold"));
    q.def("setDelta", &PruneQuantizerCell::setDelta, py::arg("delta"));
    q.def("setStartThreshold", &PruneQuantizerCell::setStartThreshold, py::arg("start"));
    q.def("setStepSizeThreshold", &PruneQuantizerCell::setStepSizeThreshold, py::arg("stepsize"));
    q.def("setGammaThreshold", &PruneQuantizerCell::setGammaThreshold, py::arg("gamma"));
    q.def("getMasksWeights", &PruneQuantizerCell::getMasksWeights, py::return_value_policy::reference);
    q.def("getPruningMode", &PruneQuantizerCell::getPruningMode);
    q.def("getThreshold", &PruneQuantizerCell::getThreshold);
    q.def("getDelta", &PruneQuantizerCell::getDelta);

    py::enum_<PruneQuantizerCell::PruningMode>(q, "PruningMode")
    .value("Identity", PruneQuantizerCell::PruningMode::Identity)
    .value("Static", PruneQuantizerCell::PruningMode::Static)
    .value("Gradual", PruneQuantizerCell::PruningMode::Gradual)
    .export_values();
}

}
