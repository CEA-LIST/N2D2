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

#include "Database/Database.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_Database(py::module &m) {
    py::class_<Database, std::shared_ptr<Database>, Parameterizable> db(m, "Database", py::multiple_inheritance());

    db.doc() = 
    R"mydelimiter(
    Database specifications:

    - Genericity: load image and sound, 1D, 2D or 3D data

    - Associate a label for each data point or global to the stimulus, 1D or 2D labels

    - ROIs handling:
    
    + Convert ROIs to data point labels

    + Extract one or multiple ROIs from an initial dataset to create as many corresponding stimuli
    )mydelimiter";

    py::enum_<Database::StimuliSet>(db, "StimuliSet")
    .value("Learn", Database::StimuliSet::Learn)
    .value("Validation", Database::StimuliSet::Validation)
    .value("Test", Database::StimuliSet::Test)
    .value("Unpartitioned", Database::StimuliSet::Unpartitioned)
    .export_values();

    py::enum_<Database::StimuliSetMask>(db, "StimuliSetMask")
    .value("LearnOnly", Database::StimuliSetMask::LearnOnly)
    .value("ValidationOnly", Database::StimuliSetMask::ValidationOnly)
    .value("TestOnly", Database::StimuliSetMask::TestOnly)
    .value("NoLearn", Database::StimuliSetMask::NoLearn)
    .value("NoValidation", Database::StimuliSetMask::NoValidation)
    .value("NoTest", Database::StimuliSetMask::NoTest)
    .value("All", Database::StimuliSetMask::All)
    .export_values();

    db.def(py::init<bool>(), py::arg("loadDataInMemory") = false)
    // .def("loadROIs", &Database::loadROIs, py::arg("fileName"), py::arg("relPath") = "", py::arg("noImageSize") = false)
    // .def("loadROIsDir", &Database::loadROIsDir, py::arg("dirName"), py::arg("fileExt") = "", py::arg("depth") = 0)
    // .def("saveROIs", &Database::saveROIs, py::arg("fileName"), py::arg("header") = "")
    // .def("logStats", &Database::logStats, py::arg("sizeFileName"), py::arg("labelFileName"), py::arg("setMask") = Database::All)
    // .def("logROIsStats", &Database::logROIsStats, py::arg("sizeFileName"), py::arg("labelFileName"), py::arg("setMask") = Database::All)
    

    .def("load", &Database::load, py::arg("dataPath"), py::arg("labelPath") = "", py::arg("extractROIs") = false,
     R"mydelimiter(
     Load data.
     
     :param dataPath: Path to the dataset file.
     :type dataPath: str
     :param labelPath: Path to the label file.
     :type labelPath: str, optional
     :param extractROIs: If True extract ROI
     :type extractROIs: bool, optional
    )mydelimiter")
    .def("getLabelName", &Database::getLabelName, py::arg("label"),
     R"mydelimiter(
     Load data.
     
     :param label: Label index.
     :type label: int
    )mydelimiter")
    ;

        // TODO : Find a better method to add description to overloaded method
    // As mentionned here https://github.com/pybind/pybind11/issues/2619 pybind + shpinx have trouble generating docstring for overloaded method.
    // The current best fix is to disable function signatures, this seems to be currently acceptable. 

    py::options options;
    options.disable_function_signatures(); // Apply only on this scope so no need to enable function signature again.

    db.def("partitionStimuli", (void (Database::*)(unsigned int, Database::StimuliSet))(&Database::partitionStimuli), py::arg("nbStimuli"), py::arg("set"));
#if SIZE_MAX != 0xFFFFFFFF
    db.def("partitionStimuli", (void (Database::*)(double, double, double))(&Database::partitionStimuli), py::arg("learn"), py::arg("validation"), py::arg("test"));
#endif

    db.def("partitionStimuliPerLabel", (void (Database::*)(unsigned int, Database::StimuliSet))(&Database::partitionStimuliPerLabel), py::arg("nbStimuliPerLabel"), py::arg("set"));
#if SIZE_MAX != 0xFFFFFFFF
    db.def("partitionStimuliPerLabel", (void (Database::*)(double, double, double, bool))(&Database::partitionStimuliPerLabel), py::arg("learnPerLabel"), py::arg("validationPerLabel"), py::arg("testPerLabel"),  py::arg("equiLabel") = false);
#endif

    db.def("getNbStimuli", (unsigned int (Database::*)() const)(&Database::getNbStimuli),
    R"mydelimiter(
    Returns the total number of loaded stimuli.
    
    :return: Number of stimuli
    :rtype: int
    )mydelimiter")
    .def("getNbStimuli", (unsigned int (Database::*)(Database::StimuliSet) const)(&Database::getNbStimuli), py::arg("set"),
     R"mydelimiter(
     Returns the number of stimuli in one stimuli set.
     
     :param set: Set of stimuli
     :type set: :py:class:`N2D2.Database.StimuliSet`
     :return: Number of stimuli in the set
     :rtype: int
    )mydelimiter")
    .def("getStimulusLabel", (int (Database::*)(Database::StimulusID) const)(&Database::getStimulusLabel), py::arg("id"),
     R"mydelimiter(
     Returns the label of a stimuli.
     
     :param set: id of stimuli
     :type set: :py:class:`N2D2.Database.StimulusID`
     :return: Label of stimuli
     :rtype: int
    )mydelimiter")
    .def("getStimulusLabel", (int (Database::*)(Database::StimuliSet, unsigned int) const)(&Database::getStimulusLabel), py::arg("set"), py::arg("index"),
     R"mydelimiter(
     Returns the label of a stimuli.
     
     :param set: Set of stimuli
     :type set: :py:class:`N2D2.Database.StimuliSet`
     :param index: Index of stimuli
     :type index: int
     :return: Label of stimuli
     :rtype: int
    )mydelimiter")
    .def("getLoadDataInMemory", (bool (Database::*)())(&Database::getLoadDataInMemory))
    .def("getStimuliDepth", &Database::getStimuliDepth)
    ;

}
}
