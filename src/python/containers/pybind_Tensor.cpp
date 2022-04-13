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

#include "containers/Tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

namespace N2D2 {
template<typename T, typename std::enable_if<!std::is_same<T, bool>::value>::type* = nullptr>
void declare_Tensor_buffer_protocol(py::class_<Tensor<T>, BaseTensor>& tensor) {
    // Buffer protocol
    tensor.def_buffer([](Tensor<T>& b) -> py::buffer_info {
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
            sizeof(T),                                  /* Size of one scalar */
            py::format_descriptor<T>::format(),         /* Python struct-style format descriptor */
            b.nbDims(),                                 /* Number of dimensions */
            dims,                                       /* Buffer dimensions */
            strides                                     /* Strides (in bytes) for each index */
        );
    })
    .def(py::init([]( py::array_t<T, py::array::c_style | py::array::forcecast> b) {
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
        return new Tensor<T>(dims, static_cast<T*>(info.ptr));
    }));
}

template<typename T, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
void declare_Tensor_buffer_protocol(py::class_<Tensor<T>, BaseTensor>& /*tensor*/) {
    // No buffer protocol for bool!
}

template<typename T>
void declare_Tensor(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("Tensor_" + typeStr);
    py::class_<Tensor<T>, BaseTensor> tensor(m, pyClassName.c_str(), py::multiple_inheritance(), py::buffer_protocol());
    tensor.def(py::init<>())
    .def(py::init<const std::vector<size_t>&, const T&>(), py::arg("dims"), py::arg("value") = T())
    /// Bare bones interface
    .def("__getitem__", [](const Tensor<T>& b, size_t i) {
        if (i >= b.size()) throw py::index_error();
        return b(i);
    })
    .def("__setitem__", [](Tensor<T>& b, size_t i, T v) {
        if (i >= b.size()) throw py::index_error();
        b(i) = v;
    })
    .def("__len__", [](BaseTensor& b) { return b.size(); })
    /// Optional sequence protocol operations
    .def("__iter__", [](const Tensor<T>& b) { return py::make_iterator(b.begin(), b.end()); },
                        py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
    .def("__contains__", [](const Tensor<T>& b, T v) {
        return (std::find(b.begin(), b.end(), v) != b.end());
    })
    .def("__reversed__", [](const Tensor<T>& b) -> Tensor<T> {
        std::vector<size_t> reversedDims(b.dims());
        std::reverse(reversedDims.begin(), reversedDims.end());

        std::vector<T> reversedData(b.begin(), b.end());
        std::reverse(reversedData.begin(), reversedData.end());

        return Tensor<T>(reversedDims,
                         reversedData.begin(), reversedData.end());
    })
    /// Slicing protocol (optional)
    .def("__getitem__", [](const Tensor<T>& b, py::slice slice) -> Tensor<T>* {
        size_t start, stop, step, slicelength;
        if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        Tensor<T>* t = new Tensor<T>({slicelength});
        for (size_t i = 0; i < slicelength; ++i) {
            (*t)(i) = b(start); start += step;
        }
        return t;
    })
    .def("__setitem__", [](Tensor<T>& b, py::slice slice, const Tensor<T>& value) {
        size_t start, stop, step, slicelength;
        if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        if (slicelength != value.size())
            throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");
        for (size_t i = 0; i < slicelength; ++i) {
            b(start) = value(i); start += step;
        }
    })
    .def("__setitem__", [](Tensor<T>& b, py::slice slice, const T& value) {
        size_t start, stop, step, slicelength;
        if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        for (size_t i = 0; i < slicelength; ++i) {
            b(start) = value; start += step;
        }
    })
    .def("__str__", [](Tensor<T>& b) { 
        std::ostringstream oss;
        oss << b;
        std::string str = oss.str();
        return str;
    })
    .def("sum", &Tensor<T>::sum, py::arg("valAbs")=false)
    .def("mean", &Tensor<T>::mean, py::arg("valAbs")=false)
    .def("fill", &Tensor<T>::fill, py::arg("value"))
    ;

    declare_Tensor_buffer_protocol(tensor);
}

void init_Tensor(py::module &m) {
    py::class_<BaseTensor>(m, "BaseTensor")
    .def("empty", &BaseTensor::empty)
    .def("dimX", &BaseTensor::dimX)
    .def("dimY", &BaseTensor::dimY)
    .def("dimD", &BaseTensor::dimD)
    .def("dimZ", &BaseTensor::dimZ)
    .def("dimB", &BaseTensor::dimB)
    .def("size", &BaseTensor::size)
    // .def("reserve", (void (BaseTensor::*)(const std::vector<size_t>&)) &BaseTensor::reserve, py::arg("dims"))
    .def("resize", (void (BaseTensor::*)(const std::vector<size_t>&)) &BaseTensor::resize, py::arg("dims"))
    .def("reshape", (void (BaseTensor::*)(const std::vector<size_t>&)) &BaseTensor::reshape, py::arg("dims"))
    .def("clear", &BaseTensor::clear)
    // .def("save", &BaseTensor::save, py::arg("data"))
    // .def("load", &BaseTensor::load, py::arg("data"))
    .def("synchronizeDToH", (void (BaseTensor::*)() const) &BaseTensor::synchronizeDToH)
    .def("synchronizeHToD", (void (BaseTensor::*)() const) &BaseTensor::synchronizeHToD)
    .def("synchronizeDToHBased", &BaseTensor::synchronizeDToHBased)
    .def("synchronizeHBasedToD", &BaseTensor::synchronizeHBasedToD)
    .def("synchronizeDBasedToH", &BaseTensor::synchronizeDBasedToH)
    .def("synchronizeHToDBased", &BaseTensor::synchronizeHToDBased)
    .def("op_assign", &BaseTensor::operator=)
    .def("nbDims", &BaseTensor::nbDims)
    .def("dims", &BaseTensor::dims)
    .def("isValid", &BaseTensor::isValid, py::arg("dev")=-1)
    .def("setValid", &BaseTensor::setValid, py::arg("dev")=-1)
    // .def("clearValid", &BaseTensor::clearValid)
    .def("getType", &BaseTensor::getType)
    .def("getTypeName", &BaseTensor::getTypeName)
    ;

    declare_Tensor<float>(m, "float");
    declare_Tensor<double>(m, "double");
    declare_Tensor<char>(m, "char");
    declare_Tensor<unsigned char>(m, "unsigned_char");
    declare_Tensor<short>(m, "short");
    declare_Tensor<int>(m, "int");
    declare_Tensor<unsigned int>(m, "unsigned_int");
    declare_Tensor<long>(m, "long");
    declare_Tensor<unsigned long>(m, "unsigned_long");
    declare_Tensor<long long>(m, "long long");
    declare_Tensor<unsigned long long>(m, "unsigned_long_long");
    declare_Tensor<bool>(m, "bool");
}
}
