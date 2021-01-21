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
#include "Transformation/ChannelExtractionTransformation.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_ChannelExtractionTransformation(py::module &m) {
    py::class_<ChannelExtractionTransformation, std::shared_ptr<ChannelExtractionTransformation>, Transformation> cet (m, "ChannelExtractionTransformation", py::multiple_inheritance());

    py::enum_<ChannelExtractionTransformation::Channel>(cet, "Channel")
    .value("Red", ChannelExtractionTransformation::Channel::Red)
    .value("Green", ChannelExtractionTransformation::Channel::Green)
    .value("Blue", ChannelExtractionTransformation::Channel::Blue)
    .value("Hue", ChannelExtractionTransformation::Channel::Hue)
    .value("Saturation", ChannelExtractionTransformation::Channel::Saturation)
    .value("Value", ChannelExtractionTransformation::Channel::Value)
    .value("Y", ChannelExtractionTransformation::Channel::Y)
    .value("Cb", ChannelExtractionTransformation::Channel::Cb)
    .value("Cr", ChannelExtractionTransformation::Channel::Cr)
    .export_values();

    cet.def(py::init<ChannelExtractionTransformation::Channel>(), py::arg("channel"));


}
}
#endif
