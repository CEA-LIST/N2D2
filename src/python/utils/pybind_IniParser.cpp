/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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

#include "utils/IniParser.hpp"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {

void init_IniParser(py::module &m) {
    py::class_<IniParser, std::shared_ptr<IniParser>> (m, "IniParser")
    .def(py::init<>())
    .def("load", (void (IniParser::*)(const std::string&)) (&IniParser::load), py::arg("fileName"))
    .def("currentSection", &IniParser::currentSection)
    .def("setProperty", (void (IniParser::*)(const std::string&, const int&)) (&IniParser::setProperty), py::arg("name"), py::arg("value"))
    .def("setProperty", (void (IniParser::*)(const std::string&, const float&)) (&IniParser::setProperty), py::arg("name"), py::arg("value"))
    .def("setProperty", (void (IniParser::*)(const std::string&, const bool&)) (&IniParser::setProperty), py::arg("name"), py::arg("value"))
    .def("setProperty", (void (IniParser::*)(const std::string&, const std::string&)) (&IniParser::setProperty), py::arg("name"), py::arg("value"))
    .def("setProperty", (void (IniParser::*)(const std::string&, const std::vector<std::string>&)) (&IniParser::setProperty), py::arg("name"), py::arg("value"))
    ;
}
}
