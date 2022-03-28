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

#include "Transformation/DistortionTransformation.hpp"
#include "Transformation/Transformation.hpp"
#include "Transformation/PadCropTransformation.hpp"
#include "Transformation/CompositeTransformation.hpp"
#include "Transformation/AffineTransformation.hpp"
#include "Transformation/ChannelExtractionTransformation.hpp"
#include "Transformation/ColorSpaceTransformation.hpp"
#include "Transformation/CompressionNoiseTransformation.hpp"
#include "Transformation/DCTTransformation.hpp"
#include "Transformation/DFTTransformation.hpp"
#include "Transformation/EqualizeTransformation.hpp"
#include "Transformation/ExpandLabelTransformation.hpp"
#include "Transformation/WallisFilterTransformation.hpp"
#include "Transformation/ThresholdTransformation.hpp"
#include "Transformation/SliceExtractionTransformation.hpp"
#include "Transformation/ReshapeTransformation.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "Transformation/RangeClippingTransformation.hpp"
#include "Transformation/RangeAffineTransformation.hpp"
#include "Transformation/RandomAffineTransformation.hpp"
#include "Transformation/NormalizeTransformation.hpp"
#include "Transformation/MorphologyTransformation.hpp"
#include "Transformation/MorphologicalReconstructionTransformation.hpp"
#include "Transformation/MagnitudePhaseTransformation.hpp"
#include "Transformation/LabelSliceExtractionTransformation.hpp"
#include "Transformation/LabelExtractionTransformation.hpp"
#include "Transformation/GradientFilterTransformation.hpp"
#include "Transformation/ApodizationTransformation.hpp"
#include "Transformation/TrimTransformation.hpp"
#include "Transformation/FilterTransformation.hpp"
#include "Transformation/FlipTransformation.hpp"
#include "Transformation/RandomResizeCropTransformation.hpp"
#include "Transformation/CustomTransformation.hpp"





#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
template<class T>
void make_compositable(py::class_<CompositeTransformation, std::shared_ptr<CompositeTransformation>, Transformation> &m) {
    m.def(py::init<const T&>(), py::arg("transformation"))
    .def("push_back", (void (CompositeTransformation::*)(const T&))(&CompositeTransformation::push_back), "Add a transformation to the list of transformation.");
    // Need an implicit conversion to be able to apply transformations with the StimuliProvider.
    py::implicitly_convertible<T, CompositeTransformation>();
    py::implicitly_convertible<CompositeTransformation, T>();

}

void init_CompositeTransformation(py::module &m) {
    py::class_<CompositeTransformation, std::shared_ptr<CompositeTransformation>, Transformation> ct(m, "CompositeTransformation", py::multiple_inheritance());
    
    ct.def(py::init<const CompositeTransformation&>(), py::arg("transformation"))
    .def("push_back", (void (CompositeTransformation::*)(const CompositeTransformation&))(&CompositeTransformation::push_back), "Add a transformation to the list of transformation.");

    make_compositable<DistortionTransformation>(ct);
    make_compositable<PadCropTransformation>(ct);
    make_compositable<AffineTransformation>(ct);
    make_compositable<ChannelExtractionTransformation>(ct);
    make_compositable<ColorSpaceTransformation>(ct);
    make_compositable<CompressionNoiseTransformation>(ct);
    make_compositable<DCTTransformation>(ct);
    make_compositable<DFTTransformation>(ct);
    make_compositable<EqualizeTransformation>(ct);
    make_compositable<ExpandLabelTransformation>(ct);
    make_compositable<FlipTransformation>(ct);
    make_compositable<WallisFilterTransformation>(ct);
    make_compositable<ThresholdTransformation>(ct);
    make_compositable<SliceExtractionTransformation>(ct);
    make_compositable<ReshapeTransformation>(ct);
    make_compositable<RescaleTransformation>(ct);
    make_compositable<RangeClippingTransformation>(ct);
    make_compositable<RangeAffineTransformation>(ct);
    make_compositable<RandomAffineTransformation>(ct);
    make_compositable<NormalizeTransformation>(ct);
    make_compositable<MorphologyTransformation>(ct);
    make_compositable<MorphologicalReconstructionTransformation>(ct);
    make_compositable<MagnitudePhaseTransformation>(ct);
    make_compositable<LabelSliceExtractionTransformation>(ct);
    make_compositable<LabelExtractionTransformation>(ct);
    make_compositable<GradientFilterTransformation>(ct);
    make_compositable<ApodizationTransformation>(ct);
    // make_compositable<TrimTransformation>(ct);
    make_compositable<FilterTransformation>(ct);
    make_compositable<RandomResizeCropTransformation>(ct);
    make_compositable<CustomTransformation>(ct);

}
}
