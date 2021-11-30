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


#ifdef PYBIND
#include "Solver/SGDSolver.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_SGDSolver(py::module &m) {
    py::class_<SGDSolver, std::shared_ptr<SGDSolver>, Solver> sdg (m, "SGDSolver", py::multiple_inheritance());
    
    py::enum_<SGDSolver::LearningRatePolicy>(sdg, "LearningRatePolicy")
    .value("None", SGDSolver::LearningRatePolicy::None)
    .value("StepDecay", SGDSolver::LearningRatePolicy::StepDecay)
    .value("ExponentialDecay", SGDSolver::LearningRatePolicy::ExponentialDecay)
    .value("InvTDecay", SGDSolver::LearningRatePolicy::InvTDecay)
    .value("PolyDecay", SGDSolver::LearningRatePolicy::PolyDecay)
    .value("InvDecay", SGDSolver::LearningRatePolicy::InvDecay)
    .value("CosineDecay", SGDSolver::LearningRatePolicy::CosineDecay)
    .export_values();
    
    sdg.def("getmLearningRate", &SGDSolver::getmLearningRate)
    .def("getMomentum", &SGDSolver::getMomentum)
    .def("getDecay", &SGDSolver::getDecay)
    .def("getMinDecay", &SGDSolver::getMinDecay)
    .def("getPower", &SGDSolver::getPower)
    .def("getIterationSize", &SGDSolver::getIterationSize)
    .def("getMaxIterations", &SGDSolver::getMaxIterations)
    .def("getWarmUpDuration", &SGDSolver::getWarmUpDuration)
    .def("getWarmUpLRFrac", &SGDSolver::getWarmUpLRFrac)
    .def("getLearningRatePolicy", &SGDSolver::getLearningRatePolicy)
    .def("getLearningRateStepSize", &SGDSolver::getLearningRateStepSize)
    .def("getLearningRateDecay", &SGDSolver::getLearningRateDecay)
    .def("getmClamping", &SGDSolver::getmClamping)
    .def("getPolyakMomentum", &SGDSolver::getPolyakMomentum)
    ;

}
}
#endif
