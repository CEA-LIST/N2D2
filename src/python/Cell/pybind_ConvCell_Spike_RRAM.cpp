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
#include "Cell/ConvCell_Spike_RRAM.hpp"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace N2D2 {
void init_ConvCell_Spike_RRAM(py::module &m) {
    py::class_<ConvCell_Spike_RRAM, std::shared_ptr<ConvCell_Spike_RRAM>, ConvCell_Spike> (m, "ConvCell_Spike_RRAM", py::multiple_inheritance())
    .def(py::init<
    Network&, 
    const DeepNet&,
    const std::string&,
    const std::vector<unsigned int>&, 
    unsigned int, 
    const std::vector<unsigned int>&, 
    const std::vector<unsigned int>&,
    const std::vector<int>&,
    const std::vector<unsigned int>&>(),
    py::arg("net"),
    py::arg("deepNet"),
    py::arg("name"),
    py::arg("kernelDims"),
    py::arg("nbOutputs"),
    py::arg("subSampleDims") = std::vector<unsigned int>(2, 1U),
    py::arg("strideDims") = std::vector<unsigned int>(2, 1U),
    py::arg("paddingDims") = std::vector<int>(2, 0),
    py::arg("dilationDims") = std::vector<unsigned int>(2, 1U))
    ;
}
}
#endif
