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
#include "Quantizer/Cell/QuantizerCell.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_QuantizerCell(py::module &m) {

    

    py::class_<QuantizerCell, std::shared_ptr<QuantizerCell>, Parameterizable> q(m, "QuantizerCell", py::multiple_inheritance()); 
    q.doc() = "QuantizerCell is the abstract base object for any kind of cell quantizer";
    q.def("addWeights", &QuantizerCell::addWeights, py::arg("weights"), py::arg("diffWeights"));
    q.def("addBiases", &QuantizerCell::addBiases, py::arg("biases"), py::arg("diffBiases"));
    q.def("setSolver", &QuantizerCell::setSolver, py::arg("solver"));
    q.def("getType", &QuantizerCell::getType);
    q.def("getQuantizedWeights", &QuantizerCell::getQuantizedWeights, py::arg("k"), py::return_value_policy::reference);
    q.def("getQuantizedBiases", &QuantizerCell::getQuantizedBiases, py::return_value_policy::reference);
    q.def("setRange", &QuantizerCell::setRange, py::arg("integerRange"));
    q.def("getRange", &QuantizerCell::getRange);
    q.def("getQuantMode", &QuantizerCell::getQuantMode);
    q.def("getDiffFullPrecisionWeights", &QuantizerCell::getDiffFullPrecisionWeights, py::arg("k"), py::return_value_policy::reference);
    q.def("getDiffFullPrecisionBiases", &QuantizerCell::getDiffFullPrecisionBiases, py::return_value_policy::reference);
    
    // For tests
    q.def("initialize", &QuantizerCell::initialize);
    q.def("propagate", &QuantizerCell::propagate);
    q.def("back_propagate", &QuantizerCell::back_propagate);
    // End for tests
    
    py::enum_<QuantizerCell::QuantMode>(q, "QuantMode")
    .value("Default", QuantizerCell::QuantMode::Default)
    .value("Symmetric", QuantizerCell::QuantMode::Symmetric)
    .value("Asymmetric", QuantizerCell::QuantMode::Asymmetric)
    .value("FullRange", QuantizerCell::QuantMode::FullRange)
    .export_values();
}
}

#endif

