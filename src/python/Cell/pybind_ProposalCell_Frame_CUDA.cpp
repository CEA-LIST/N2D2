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

#include "Cell/ProposalCell_Frame_CUDA.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#if SIZE_MAX != 0xFFFFFFFF

namespace N2D2 {
void init_ProposalCell_Frame_CUDA(py::module &m) {
    py::class_<ProposalCell_Frame_CUDA, std::shared_ptr<ProposalCell_Frame_CUDA>, ProposalCell,  Cell_Frame_CUDA<Float_T>> (m, "ProposalCell_Frame_CUDA", py::multiple_inheritance()) 
    .def(py::init<
    const DeepNet&, 
    const std::string&,
    StimuliProvider&, 
    const unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    bool,
    std::vector<double>,
    std::vector<double>,
    std::vector<unsigned int>,
    std::vector<unsigned int>
    >(),
    py::arg("deepNet"),
    py::arg("name"),
    py::arg("sp"),
    py::arg("nbOutputs"),
    py::arg("nbProposals"),
    py::arg("scoreIndex") = 0,
    py::arg("IoUIndex") = 5,
    py::arg("isNms") = false,
    py::arg("meansFactor") = std::vector<double>(),
    py::arg("stdFactor") = std::vector<double>(),
    py::arg("numParts") = std::vector<unsigned int>(),
    py::arg("numTemplates") = std::vector<unsigned int>()
    );
#endif
}
}

#endif

