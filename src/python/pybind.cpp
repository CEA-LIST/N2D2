/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#ifdef PYBIND
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
void init_CudaContext(py::module&);
void init_Tensor(py::module&);
void init_CudaTensor(py::module&);
void init_Network(py::module&);
void init_Database(py::module&);
void init_StimuliProvider(py::module&);
void init_Cell(py::module&);
void init_Cell_Frame_Top(py::module&);
void init_Cell_Frame(py::module&);
void init_Cell_Frame_CUDA(py::module&);
void init_Target(py::module&);
void init_TargetScore(py::module&);
void init_DeepNet(py::module&);
void init_DeepNetGenerator(py::module&);

PYBIND11_MODULE(N2D2, m) {
    init_CudaContext(m);
    init_Tensor(m);
    init_CudaTensor(m);
    init_Network(m);
    init_Database(m);
    init_StimuliProvider(m);
    init_Cell(m);
    init_Cell_Frame_Top(m);
    init_Cell_Frame(m);
    init_Cell_Frame_CUDA(m);
    init_Target(m);
    init_TargetScore(m);
    init_DeepNet(m);
    init_DeepNetGenerator(m);
}
}

#endif
