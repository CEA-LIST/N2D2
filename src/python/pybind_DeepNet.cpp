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

#include "Xnet/Monitor.hpp"
#include "CMonitor.hpp"
#include "StimuliProvider.hpp"
#include "DeepNet.hpp"
#include "Cell/Cell_Frame_Top.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_DeepNet(py::module &m) {
    py::class_<DeepNet, std::shared_ptr<DeepNet>, Parameterizable>(m, "DeepNet", py::multiple_inheritance())
    .def(py::init<Network&>(), py::arg("net"))
    .def("addCell", &DeepNet::addCell, py::arg("cell"), py::arg("parents"))
    .def("removeCell", &DeepNet::removeCell, py::arg("cell"), py::arg("reconnect") = true)
    .def("addTarget", &DeepNet::addTarget, py::arg("target"))
    .def("addMonitor", &DeepNet::addMonitor, py::arg("name"), py::arg("monitor"))
    .def("addCMonitor", &DeepNet::addCMonitor, py::arg("name"), py::arg("monitor"))
    .def("update", (std::vector<std::pair<std::string, long long unsigned int>> (DeepNet::*)(bool, Time_T, Time_T, bool)) &DeepNet::update, py::arg("log"), py::arg("start"), py::arg("stop") = 0, py::arg("update") = true)
    .def("save", &DeepNet::save, py::arg("dirName"))
    .def("load", &DeepNet::load, py::arg("dirName"))
    .def("saveNetworkParameters", &DeepNet::saveNetworkParameters)
    .def("loadNetworkParameters", &DeepNet::loadNetworkParameters)
    .def("exportNetworkFreeParameters", &DeepNet::exportNetworkFreeParameters, py::arg("dirName"))
    .def("exportNetworkSolverParameters", &DeepNet::exportNetworkSolverParameters, py::arg("dirName"))
    .def("importNetworkFreeParameters", (void (DeepNet::*)(const std::string&, bool)) &DeepNet::importNetworkFreeParameters, py::arg("dirName"), py::arg("ignoreNotExists") = false)
    .def("importNetworkFreeParameters", (void (DeepNet::*)(const std::string&, const std::string&)) &DeepNet::importNetworkFreeParameters, py::arg("dirName"), py::arg("weightName"))
    //.def("importNetworkSolverParameters", &DeepNet::importNetworkSolverParameters, py::arg("dirName"))
    // .def("checkGradient", &DeepNet::checkGradient, py::arg("epsilon") = 1.0e-4, py::arg("maxError") = 1.0e-6)
    .def("initialize", &DeepNet::initialize)
    // .def("learn", &DeepNet::learn, py::arg("timings") = NULL)
    .def("test", &DeepNet::test, py::arg("set"), py::arg("timings") = NULL)
    .def("propagate", (void (DeepNet::*)(bool)) &DeepNet::propagate, py::arg("inference"))
    .def("propagate", (void (DeepNet::*)(Database::StimuliSet, bool, std::vector<std::pair<std::string, double> >*)) &DeepNet::propagate, py::arg("set"), py::arg("inference"), py::arg("timings") = NULL)
    .def("backPropagate", &DeepNet::backPropagate, py::arg("timings") = std::vector<std::pair<std::string, double> >())
    .def("update", (void (DeepNet::*)(std::vector<std::pair<std::string, double> >*)) &DeepNet::update, py::arg("timings") = std::vector<std::pair<std::string, double> >())
    .def("cTicks", &DeepNet::cTicks, py::arg("start"), py::arg("stop"), py::arg("timestep"), py::arg("record") = false)
    .def("cTargetsProcess", &DeepNet::cTargetsProcess, py::arg("set"))
    .def("cReset", &DeepNet::cReset, py::arg("timestamp") = 0)
    .def("initializeCMonitors", &DeepNet::initializeCMonitors, py::arg("nbTimesteps"))
    .def("spikeCodingCompare", &DeepNet::spikeCodingCompare, py::arg("dirName"), py::arg("idx"))
    .def("fuseBatchNorm", &DeepNet::fuseBatchNorm)
    .def("removeDropout", &DeepNet::removeDropout)
    .def("setDatabase", &DeepNet::setDatabase, py::arg("database"))
    .def("setStimuliProvider", &DeepNet::setStimuliProvider, py::arg("sp"))
    .def("getDatabase", &DeepNet::getDatabase)
    .def("getStimuliProvider", &DeepNet::getStimuliProvider)
#ifdef CUDA
    .def("getMasterDevice", &DeepNet::getMasterDevice)
#endif
    .def("getCells", &DeepNet::getCells)
    .def("getCell", &DeepNet::getCell<Cell>, py::arg("name"))
    .def("getCell_Frame_Top", &DeepNet::getCell<Cell_Frame_Top>, py::arg("name"))
    .def("getMonitor", &DeepNet::getMonitor, py::arg("name"))
    .def("getCMonitor", &DeepNet::getCMonitor, py::arg("name"))
    .def("getLayers", &DeepNet::getLayers)
    // .def("getLayer", &DeepNet::getLayer, py::arg("layer"))
    .def("getChildCells", &DeepNet::getChildCells, py::arg("name"))
    .def("getParentCells", &DeepNet::getParentCells, py::arg("name"))
    .def("getTargets", &DeepNet::getTargets)
    .def("getNetwork", &DeepNet::getNetwork)
    .def("getName", &DeepNet::getName)
    // .def("getStats", &DeepNet::getStats)
    //.def("getReceptiveField", &DeepNet::getReceptiveField, py::arg("name"), py::arg("outputField") = std::vector<unsigned int>())
    // .def("clearAll", &DeepNet::clearAll)
    // .def("clearActivity", &DeepNet::clearActivity)
    // .def("clearFiringRate", &DeepNet::clearFiringRate)
    // .def("clearSuccess", &DeepNet::clearSuccess)
    // .def("clear", &DeepNet::clear, py::arg("set"))
    // .def("logOutputs", &DeepNet::logOutputs, py::arg("dirName"), py::arg("batchPos") = 0)
    // .def("logDiffInputs", &DeepNet::logDiffInputs, py::arg("dirName"), py::arg("batchPos") = 0)
    .def("logFreeParameters", &DeepNet::logFreeParameters, py::arg("dirName"))
    // .def("logSchedule", &DeepNet::logSchedule, py::arg("dirName"))
    .def("logStats", &DeepNet::logStats, py::arg("dirName"))
    // .def("logSpikeStats", &DeepNet::logSpikeStats, py::arg("dirName"), py::arg("nbPatterns"))
    .def("log", &DeepNet::log, py::arg("baseName"), py::arg("set"))
    .def("logLabelsMapping", &DeepNet::logLabelsMapping, py::arg("fileName"), py::arg("withStats") = false)
    .def("logEstimatedLabels", &DeepNet::logEstimatedLabels, py::arg("dirName"))
    .def("logEstimatedLabelsJSON", &DeepNet::logEstimatedLabelsJSON, py::arg("dirName"))
    .def("logLabelsLegend", &DeepNet::logLabelsLegend, py::arg("fileName"))
    .def("logTimings", &DeepNet::logTimings, py::arg("fileName"), py::arg("timings"))
    .def("logReceptiveFields", &DeepNet::logReceptiveFields, py::arg("fileName"))
    .def("exportNetworkFreeParameters", &DeepNet::exportNetworkFreeParameters, py::arg("dirName"))
    ;

}
}

