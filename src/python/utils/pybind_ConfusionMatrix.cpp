/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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
#include "utils/ConfusionMatrix.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {

void init_ConfusionMatrix(py::module &m) {

  
    py::enum_<ConfusionTableMetric>(m, "ConfusionTableMetric")
    .value("Sensitivity", ConfusionTableMetric::Sensitivity)
    .value("Specificity", ConfusionTableMetric::Specificity)
    .value("Precision", ConfusionTableMetric::Precision)
    .value("NegativePredictiveValue", ConfusionTableMetric::NegativePredictiveValue)
    .value("MissRate", ConfusionTableMetric::MissRate)
    .value("FallOut", ConfusionTableMetric::FallOut)
    .value("FalseDiscoveryRate", ConfusionTableMetric::FalseDiscoveryRate)
    .value("FalseOmissionRate", ConfusionTableMetric::FalseOmissionRate)
    .value("Accuracy", ConfusionTableMetric::Accuracy)
    .value("F1Score", ConfusionTableMetric::F1Score)
    .value("Informedness", ConfusionTableMetric::Informedness)
    .value("Markedness", ConfusionTableMetric::Markedness)
    .export_values();
}
}
#endif



 