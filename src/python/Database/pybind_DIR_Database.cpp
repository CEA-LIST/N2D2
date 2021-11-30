
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
#include "Database/DIR_Database.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_DIR_Database(py::module &m) {
    py::class_<DIR_Database, std::shared_ptr<DIR_Database>, Database>(m, "DIR_Database")
    .def(py::init<bool>(), py::arg("loadDataInMemory")=false)
    .def("loadDir", &DIR_Database::loadDir, 
        py::arg("dataPath"), 
        py::arg("depth") = 0, 
        py::arg("labelName") = "", 
        py::arg("labelDepth") = 0,
        R"mydelimiter(
        :param dataPath: path to the data
        :type dataPath: str
        :param depth: depth = 0: load stimuli only from the current directory (dirPath); depth = 1: load stimuli from dirPath and stimuli contained in the sub-directories of dirPath; depth < 0: load stimuli recursively from dirPath and all its sub-directories
        :type depth: int
        :param labelName: path to the data
        :type labelName: str
        :param labelDepth: labelDepth = -1: no label for all stimuli (label ID = -1); labelDepth = 0: uses @p labelName string for all stimuli; labelDepth = 1: uses @p labelName string for stimuli in the current; directory (dirPath) and @p labelName + sub-directory name for stimuli in the sub-directories
        :type labelDepth: int
        )mydelimiter")
    .def("setIgnoreMasks", &DIR_Database::setIgnoreMasks, py::arg("ignoreMasks"),
        R"mydelimiter(
        :param ignoreMasks: space-separated list of mask strings to ignore. If any is present in a file path, the file gets ignored. The usual * and + wildcards are allowed.
        :type ignoreMasks: list
        )mydelimiter")
    .def("setValidExtensions", &DIR_Database::setValidExtensions, py::arg("validExtensions"))
    ;
}
}
#endif