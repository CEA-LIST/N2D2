/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Cyril MOINEAU (cyril.moineau@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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

#include "Xnet/Network.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_Network(py::module &m) {
    py::class_<Network>(m, "Network")
    .def(py::init<unsigned int, bool, bool>(), py::arg("seed") = 0, py::arg("saveSeed") = true, py::arg("printTimeElapsed") = true)
    // .def("run", &Network::run, py::arg("stop") = 0, py::arg("clearActivity") = true)
    // .def("stop", &Network::stop, py::arg("stop") = 0, py::arg("discard") = false)
    // .def("reset", &Network::reset, py::arg("timestamp") = 0)
    // .def("save", &Network::save, py::arg("dirName"))
    // .def("load", &Network::load, py::arg("dirName"))
    // .def("getSpikeRecording", (const std::unordered_map<NodeId_T, NodeEvents_T>& (Network::*)()) &Network::getSpikeRecording)
    // .def("getSpikeRecording", (const NodeEvents_T& (Network::*)(NodeId_T)) &Network::getSpikeRecording, py::arg("nodeId"))
    // .def("getFirstEvent", &Network::getFirstEvent)
    // .def("getLastEvent", &Network::getLastEvent)
    // .def("getLoadSavePath", &Network::getLoadSavePath)
    ;
    
    py::class_<NetworkObserver>(m, "NetworkObserver");}
}

