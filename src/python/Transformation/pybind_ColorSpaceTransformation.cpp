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

#ifdef PYBIND
#include "Transformation/ColorSpaceTransformation.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_ColorSpaceTransformation(py::module &m) {
    py::class_<ColorSpaceTransformation, std::shared_ptr<ColorSpaceTransformation>, Transformation> cst (m, "ColorSpaceTransformation", py::multiple_inheritance());

    py::enum_<ColorSpaceTransformation::ColorSpace>(cst, "ColorSpace")
    .value("BGR", ColorSpaceTransformation::ColorSpace::BGR)
    .value("RGB", ColorSpaceTransformation::ColorSpace::RGB)
    .value("HSV", ColorSpaceTransformation::ColorSpace::HSV)
    .value("HLS", ColorSpaceTransformation::ColorSpace::HLS)
    .value("YCrCb", ColorSpaceTransformation::ColorSpace::YCrCb)
    .value("CIELab", ColorSpaceTransformation::ColorSpace::CIELab)
    .value("CIELuv", ColorSpaceTransformation::ColorSpace::CIELuv)
    .value("RGB_to_BGR", ColorSpaceTransformation::ColorSpace::RGB_to_BGR)
    .value("RGB_to_HSV", ColorSpaceTransformation::ColorSpace::RGB_to_HSV)
    .value("RGB_to_HLS", ColorSpaceTransformation::ColorSpace::RGB_to_HLS)
    .value("RGB_to_YCrCb", ColorSpaceTransformation::ColorSpace::RGB_to_YCrCb)
    .value("RGB_to_CIELab", ColorSpaceTransformation::ColorSpace::RGB_to_CIELab)
    .value("RGB_to_CIELuv", ColorSpaceTransformation::ColorSpace::RGB_to_CIELuv)
    .value("HSV_to_BGR", ColorSpaceTransformation::ColorSpace::HSV_to_BGR)
    .value("HSV_to_RGB", ColorSpaceTransformation::ColorSpace::HSV_to_RGB)
    .value("HLS_to_BGR", ColorSpaceTransformation::ColorSpace::HLS_to_BGR)
    .value("HLS_to_RGB", ColorSpaceTransformation::ColorSpace::HLS_to_RGB)
    .value("YCrCb_to_BGR", ColorSpaceTransformation::ColorSpace::YCrCb_to_BGR)
    .value("YCrCb_to_RGB", ColorSpaceTransformation::ColorSpace::YCrCb_to_RGB)
    .value("CIELab_to_BGR", ColorSpaceTransformation::ColorSpace::CIELab_to_BGR)
    .value("CIELab_to_RGB", ColorSpaceTransformation::ColorSpace::CIELab_to_RGB)
    .value("CIELuv_to_BGR", ColorSpaceTransformation::ColorSpace::CIELuv_to_BGR)
    .value("CIELuv_to_RGB", ColorSpaceTransformation::ColorSpace::CIELuv_to_RGB)
    .export_values();

    cst.def(py::init<ColorSpaceTransformation::ColorSpace>(), py::arg("colorSpace"));


}
}
#endif

#endif
