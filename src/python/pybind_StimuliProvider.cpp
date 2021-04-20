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
#include "StimuliProvider.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_StimuliProvider(py::module &m) {
    py::class_<StimuliProvider, std::shared_ptr<StimuliProvider>>(m, "StimuliProvider", py::multiple_inheritance())
    .def(py::init<Database&, const std::vector<size_t>&, unsigned int, bool>(), py::arg("database"), py::arg("size"), py::arg("batchSize") = 1, py::arg("compositeStimuli") = false)
    .def("cloneParameters", &StimuliProvider::cloneParameters)
    .def("logTransformations", &StimuliProvider::logTransformations, py::arg("fileName"), py::arg("setMask"))
    .def("future", &StimuliProvider::future)
    .def("synchronize", &StimuliProvider::synchronize)
    .def("getRandomIndex", &StimuliProvider::getRandomIndex, py::arg("set"))
    .def("getRandomID", &StimuliProvider::getRandomID, py::arg("set"))
    .def("readRandomBatch", &StimuliProvider::readRandomBatch, py::arg("set"))
    .def("readRandomStimulus", &StimuliProvider::readRandomStimulus, py::arg("set"), py::arg("batchPos") = 0, py::arg("dev") = -1)
    .def("readBatch", (void (StimuliProvider::*)(Database::StimuliSet, unsigned int)) &StimuliProvider::readBatch, py::arg("set"), py::arg("startIndex"))
    .def("readBatch", (void (StimuliProvider::*)(Database::StimuliSet)) &StimuliProvider::readBatch, py::arg("set"))
    .def("streamBatch", &StimuliProvider::streamBatch, py::arg("startIndex") = -1, py::arg("dev") = -1)
    .def("readStimulusBatch", (void (StimuliProvider::*)(Database::StimulusID, Database::StimuliSet, int)) &StimuliProvider::readStimulusBatch, py::arg("id"), py::arg("set"), py::arg("dev") = -1)
    .def("readStimulusBatch", (Database::StimulusID (StimuliProvider::*)(Database::StimuliSet, unsigned int, int)) &StimuliProvider::readStimulusBatch, py::arg("set"), py::arg("index"), py::arg("dev") = -1)
    .def("readStimulus", (void (StimuliProvider::*)(Database::StimulusID, Database::StimuliSet, unsigned int, int)) &StimuliProvider::readStimulus, py::arg("id"), py::arg("set"), py::arg("batchPos") = 0, py::arg("dev") = -1)
    .def("readStimulus", (Database::StimulusID (StimuliProvider::*)(Database::StimuliSet, unsigned int, unsigned int, int)) &StimuliProvider::readStimulus, py::arg("set"), py::arg("index"), py::arg("batchPos") = 0, py::arg("dev") = -1)
    .def("readRawData", (Tensor<Float_T> (StimuliProvider::*)(Database::StimulusID) const) &StimuliProvider::readRawData, py::arg("id"))
    .def("readRawData", (Tensor<Float_T> (StimuliProvider::*)(Database::StimuliSet, unsigned int) const) &StimuliProvider::readRawData, py::arg("set"), py::arg("index"))
    .def("setBatchSize", &StimuliProvider::setBatchSize, py::arg("batchSize"))
    .def("setCachePath", &StimuliProvider::setCachePath, py::arg("path") = "")
    .def("getDatabase", (Database& (StimuliProvider::*)()) &StimuliProvider::getDatabase)
    .def("getSize", &StimuliProvider::getSize)
    // .def("getSizeX", &StimuliProvider::getSizeX)
    // .def("getSizeY", &StimuliProvider::getSizeY)
    // .def("getSizeD", &StimuliProvider::getSizeD)
    .def("getBatchSize", &StimuliProvider::getBatchSize)
    // .def("isCompositeStimuli", &StimuliProvider::isCompositeStimuli)
    .def("getNbChannels", &StimuliProvider::getNbChannels)
    .def("getNbTransformations", &StimuliProvider::getNbTransformations, py::arg("set"))
    .def("getTransformation", &StimuliProvider::getTransformation, py::arg("set"))
    .def("getOnTheFlyTransformation", &StimuliProvider::getOnTheFlyTransformation, py::arg("set"))
    .def("getChannelTransformation", &StimuliProvider::getChannelTransformation, py::arg("channel"), py::arg("set"))
    .def("getChannelOnTheFlyTransformation", &StimuliProvider::getChannelOnTheFlyTransformation, py::arg("channel"), py::arg("set"))
    .def("getBatch", &StimuliProvider::getBatch)
    .def("getData", (StimuliProvider::TensorData_T& (StimuliProvider::*)(int)) &StimuliProvider::getData, py::arg("dev") = -1)
    .def("getLabelsData", (Tensor<int>& (StimuliProvider::*)(int)) &StimuliProvider::getLabelsData, py::arg("dev") = -1)
    .def("getLabelsROIs", (const std::vector<std::vector<std::shared_ptr<ROI> > >& (StimuliProvider::*)() const) &StimuliProvider::getLabelsROIs)
    .def("getDataChannel", (const StimuliProvider::TensorData_T (StimuliProvider::*)(unsigned int, unsigned int, int) const) &StimuliProvider::getDataChannel, py::arg("channel"), py::arg("batchPos") = 0, py::arg("dev") = -1)
    .def("getLabelsDataChannel", (const Tensor<int> (StimuliProvider::*)(unsigned int, unsigned int) const) &StimuliProvider::getLabelsDataChannel, py::arg("channel"), py::arg("batchPos") = 0)
    .def("getLabelsROIs", (const std::vector<std::shared_ptr<ROI> >& (StimuliProvider::*)(unsigned int) const) &StimuliProvider::getLabelsROIs, py::arg("batchPos") = 0)
    .def("getCachePath", &StimuliProvider::getCachePath)
    .def("addTransformation", &StimuliProvider::addTransformation, py::arg("transformation"), py::arg("setMask"), "Add global CACHEABLE transformations, before applying any channel transformation")
    .def("addOnTheFlyTransformation", &StimuliProvider::addOnTheFlyTransformation, py::arg("transformation"), py::arg("setMask"), "Add global ON-THE-FLY transformations, before applying any channel transformation. The order of transformations is: global CACHEABLE, then global ON-THE-FLY")
    .def("setStreamedTensor", &StimuliProvider::setStreamedTensor, py::arg("streamedTensor"), "Set Tensor that is streamed directly to connected Deepnet");
    
    ;

}
}
#endif
