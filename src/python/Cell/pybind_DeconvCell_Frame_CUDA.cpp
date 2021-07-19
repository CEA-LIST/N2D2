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
#include "Cell/DeconvCell_Frame_CUDA.hpp"
#include "Activation/TanhActivation_Frame_CUDA.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_DeconvCell_Frame_CUDA(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("DeconvCell_Frame_CUDA_" + typeStr);
    py::class_<DeconvCell_Frame_CUDA<T>, std::shared_ptr<DeconvCell_Frame_CUDA<T>>, DeconvCell, Cell_Frame_CUDA<T>> (m, pyClassName.c_str(), py::multiple_inheritance()) 
    .def(py::init<
    const DeepNet&, 
    const std::string&, 
    const std::vector<unsigned int>&, 
    unsigned int, 
    const std::vector<unsigned int>&, 
    const std::vector<int>&, 
    const std::vector<unsigned int>&, 
    const std::shared_ptr<Activation>&>(),
         py::arg("deepNet"),
         py::arg("name"), 
         py::arg("kernelDims"), 
         py::arg("nbOutputs"), 
         py::arg("strideDims") = std::vector<unsigned int>(2, 1U), 
         py::arg("paddingDims") = std::vector<int>(2, 0), 
         py::arg("dilationDims") = std::vector<unsigned int>(2, 1U),
         py::arg("activation") = std::shared_ptr<Activation>());

}

void init_DeconvCell_Frame_CUDA(py::module &m) {
    declare_DeconvCell_Frame_CUDA<float>(m, "float");
    declare_DeconvCell_Frame_CUDA<double>(m, "double");
}
}
#endif

#endif
