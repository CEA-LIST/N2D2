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

#include "containers/CudaTensor.hpp"

#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_CudaDeviceTensor(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("CudaDeviceTensor_" + typeStr);
    py::class_<CudaDeviceTensor<T>, CudaBaseDeviceTensor>(m, pyClassName.c_str(), py::multiple_inheritance())
    .def("fill", &CudaDeviceTensor<T>::fill, py::arg("value"))
    .def("getDevicePtr", &CudaDeviceTensor<T>::getDevicePtr)
    .def("setDevicePtr", &CudaDeviceTensor<T>::setDevicePtr, py::arg("dataDevice"))
    .def("isOwner", &CudaDeviceTensor<T>::isOwner)
    .def("getType", &CudaDeviceTensor<T>::getType);
}

template<typename T>
void declare_CudaTensor(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("CudaTensor_" + typeStr);
    py::class_<CudaTensor<T>, Tensor<T>, CudaBaseTensor, BaseTensor>(m, pyClassName.c_str(), py::multiple_inheritance(), py::buffer_protocol())
    .def(py::init<>())
    .def(py::init<const std::vector<size_t>&, const T&>(), py::arg("dims"), py::arg("value") = T())
    .def("empty", [](BaseTensor& b) { return b.empty(); })
    .def("dimX", [](BaseTensor& b) { return b.dimX(); })
    .def("dimY", [](BaseTensor& b) { return b.dimY(); })
    .def("dimD", [](BaseTensor& b) { return b.dimD(); })
    .def("dimZ", [](BaseTensor& b) { return b.dimZ(); })
    .def("dimB", [](BaseTensor& b) { return b.dimB(); })
    .def("size", [](BaseTensor& b) { return b.size(); })
    .def("reserve", [](BaseTensor& b, const std::vector<size_t>& dims) { return b.reserve(dims); }, py::arg("dims"))
    .def("resize", [](BaseTensor& b, const std::vector<size_t>& dims) { return b.resize(dims); }, py::arg("dims"))
    .def("reshape", [](BaseTensor& b, const std::vector<size_t>& dims) { return b.reshape(dims); }, py::arg("dims"))
    .def("clear", [](BaseTensor& b) { return b.clear(); })
    .def("save", [](BaseTensor& b, std::ostream& data) { return b.save(data); }, py::arg("data"))
    .def("load", [](BaseTensor& b, std::istream& data) { return b.load(data); }, py::arg("data"))
    .def("synchronizeDToH", [](BaseTensor& b) { return b.synchronizeDToH(); })
    .def("synchronizeHToD", [](BaseTensor& b) { return b.synchronizeHToD(); })
    .def("synchronizeDToHBased", [](BaseTensor& b) { return b.synchronizeDToHBased(); })
    .def("synchronizeHBasedToD", [](BaseTensor& b) { return b.synchronizeHBasedToD(); })
    .def("synchronizeDBasedToH", [](BaseTensor& b) { return b.synchronizeDBasedToH(); })
    .def("synchronizeHToDBased", [](BaseTensor& b) { return b.synchronizeHToDBased(); })
    .def("nbDims", [](BaseTensor& b) { return b.nbDims(); })
    .def("dims", [](BaseTensor& b) { return b.dims(); })
    .def("isValid", [](BaseTensor& b) { return b.isValid(); })
    .def("setValid", [](BaseTensor& b) { return b.setValid(); })
    .def("clearValid", [](BaseTensor& b) { return b.clearValid(); })
    .def("getType", [](BaseTensor& b) { return b.getType(); })
    /// Bare bones interface
    .def("__getitem__", [](const CudaTensor<T>& b, size_t i) {
        if (i >= b.size()) throw py::index_error();
        return b(i);
    })
    .def("__setitem__", [](CudaTensor<T>& b, size_t i, T v) {
        if (i >= b.size()) throw py::index_error();
        b(i) = v;
    })
    .def("__len__", [](BaseTensor& b) { return b.size(); })
    /// Optional sequence protocol operations
    .def("__iter__", [](const CudaTensor<T>& b) { return py::make_iterator(b.begin(), b.end()); },
                        py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
    .def("__contains__", [](const CudaTensor<T>& b, T v) {
        return (std::find(b.begin(), b.end(), v) != b.end());
    })
    .def("__reversed__", [](const CudaTensor<T>& b) -> CudaTensor<T> {
        std::vector<size_t> reversedDims(b.dims());
        std::reverse(reversedDims.begin(), reversedDims.end());

        std::vector<T> reversedData(b.begin(), b.end());
        std::reverse(reversedData.begin(), reversedData.end());

        return CudaTensor<T>(Tensor<T>(reversedDims,
                         reversedData.begin(), reversedData.end()));
    })
    /// Slicing protocol (optional)
    .def("__getitem__", [](const Tensor<T>& b, py::slice slice) -> CudaTensor<T>* {
        size_t start, stop, step, slicelength;
        if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        CudaTensor<T>* t = new CudaTensor<T>({slicelength});
        for (size_t i = 0; i < slicelength; ++i) {
            (*t)(i) = b(start); start += step;
        }
        return t;
    })
    .def("__setitem__", [](CudaTensor<T>& b, py::slice slice, const CudaTensor<T>& value) {
        size_t start, stop, step, slicelength;
        if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        if (slicelength != value.size())
            throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");
        for (size_t i = 0; i < slicelength; ++i) {
            b(start) = value(i); start += step;
        }
    })
    .def("__setitem__", [](CudaTensor<T>& b, py::slice slice, const T& value) {
        size_t start, stop, step, slicelength;
        if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        for (size_t i = 0; i < slicelength; ++i) {
            b(start) = value; start += step;
        }
    })
    // Buffer protocol
    .def_buffer([](CudaTensor<T>& b) -> py::buffer_info {
        //assert(mData.unique());

        std::vector<ssize_t> dims;
        std::vector<ssize_t> strides;
        ssize_t stride = sizeof(T);

        for (unsigned int dim = 0; dim < b.nbDims(); ++dim) {
            dims.push_back(b.dims()[dim]);
            strides.push_back(stride);
            stride *= b.dims()[dim];
        }

        std::reverse(dims.begin(), dims.end());
        std::reverse(strides.begin(), strides.end());

        return py::buffer_info(
            &b.data()[0],                               /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
            b.nbDims(),                                      /* Number of dimensions */
            dims,                 /* Buffer dimensions */
            strides             /* Strides (in bytes) for each index */
        );
    })
    .def("__init__", [](CudaTensor<T>& m, py::array_t<T, py::array::c_style | py::array::forcecast> b) {
        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();
/*
        // Some sanity checks... -> not needed with py::array_t<...>
        if (info.format != py::format_descriptor<T>::format())
            throw std::runtime_error("Incompatible format!");

        ssize_t stride = sizeof(T);

        for (unsigned int dim = 0; dim < b.ndim; ++dim) {
            if (stride != info.strides[dim])
                throw std::runtime_error("Incompatible buffer stride!");

            stride *= info.shape[dim];
        }
*/
        const std::vector<size_t> dims(info.shape.begin(), info.shape.end());
        new (&m) CudaTensor<T>(Tensor<T>(dims, static_cast<T*>(info.ptr)));
    });
}

void init_CudaTensor(py::module &m) {
    py::class_<CudaBaseDeviceTensor>(m, "CudaBaseDeviceTensor");
    declare_CudaDeviceTensor<float>(m, "float");
    declare_CudaDeviceTensor<double>(m, "double");
    py::class_<CudaBaseTensor>(m, "CudaBaseTensor");
    declare_CudaTensor<float>(m, "float");
    declare_CudaTensor<double>(m, "double");
    declare_CudaTensor<char>(m, "char");
    declare_CudaTensor<unsigned char>(m, "unsigned char");
    declare_CudaTensor<short>(m, "short");
    declare_CudaTensor<int>(m, "int");
    declare_CudaTensor<unsigned int>(m, "unsigned int");
    declare_CudaTensor<unsigned long long>(m, "unsigned long long");
}
}
#endif
