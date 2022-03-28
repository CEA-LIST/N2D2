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


#include "Cell/ObjectDetCell_Frame.hpp"
#include "Cell/AnchorCell_Frame_Kernels_struct.hpp"
#include "StimuliProvider.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_ObjectDetCell_Frame(py::module &m) {
    py::class_<ObjectDetCell_Frame, std::shared_ptr<ObjectDetCell_Frame>, ObjectDetCell,  Cell_Frame<Float_T>> (m, "ObjectDetCell_Frame", py::multiple_inheritance()) 
    .def(py::init<
    const DeepNet&, 
    const std::string&,
    StimuliProvider&,
    const unsigned int,
    unsigned int,
    const AnchorCell_Frame_Kernels::Format,
    const AnchorCell_Frame_Kernels::PixelFormat,
    unsigned int,
    unsigned int,
    Float_T,
    std::vector<Float_T>,
    std::vector<unsigned int>,
    std::vector<unsigned int>,
    const std::vector<AnchorCell_Frame_Kernels::Anchor>&>(),
    py::arg("deepNet"),
    py::arg("name"),
    py::arg("sp"),
    py::arg("nbOutputs"),
    py::arg("nbAnchors"),
    py::arg("inputFormat"),
    py::arg("pixelFormat"),
    py::arg("nbProposals"),
    py::arg("nbClass"),
    py::arg("nmsThreshold") = 0.5,
    py::arg("scoreThreshold") = std::vector<Float_T>(1, 0.5),
    py::arg("numParts") = std::vector<unsigned int>(),
    py::arg("numTemplates") = std::vector<unsigned int>(),
    py::arg("anchors") = std::vector<AnchorCell_Frame_Kernels::Anchor>()        
    );
}
}


