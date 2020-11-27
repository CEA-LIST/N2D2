/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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
#include "Transformation/CompositeTransformation.hpp"
#include "Transformation/DistortionTransformation.hpp"
#include "Transformation/PadCropTransformation.hpp"
#include "Transformation/Transformation.hpp"



#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
template<class T>
void declare_constructor(py::class_<CompositeTransformation, std::shared_ptr<CompositeTransformation>, Transformation> &m) {
    // Templated declaration of the constructor because Transformation can't be instancied thus we have to make a constructor for every Transformation class (Possibility to use a script to generate the code). Or find another workaround. 
    m.def(py::init<const T&>(), py::arg("transformation"));
}

void init_CompositeTransformation(py::module &m) {
    py::class_<CompositeTransformation, std::shared_ptr<CompositeTransformation>, Transformation> ct(m, "CompositeTransformation", py::multiple_inheritance());
    declare_constructor<DistortionTransformation>(ct);
    declare_constructor<PadCropTransformation>(ct);

    // Trying to do an implicit conversion but it doesn't solve the problem
    // py::implicitly_convertible<Transformation, CompositeTransformation>();

}
}
#endif

#endif
