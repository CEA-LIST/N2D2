/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/


#ifdef PYBIND
#include "Quantizer/Activation/QuantizerActivation.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_QuantizerActivation(py::module &m) {
    py::class_<QuantizerActivation, std::shared_ptr<QuantizerActivation>, Parameterizable> q(m, "QuantizerActivation", py::multiple_inheritance());
    q.doc() = "QuantizerActivation is the abstract base object for any kind of activation quantizer";
    q.def("setSolver", &QuantizerActivation::setSolver, py::arg("solver"));
    q.def("getType", &QuantizerActivation::getType);
    //.def("getQuantizedActivations", &QuantizerActivation::getQuantizedActivations) // Not in cpp object at the moment
    q.def("setRange", &QuantizerActivation::setRange, py::arg("integerRange"));
}
}

#endif
