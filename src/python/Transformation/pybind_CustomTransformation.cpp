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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Transformation/CustomTransformation.hpp"
namespace py = pybind11;

namespace N2D2 {

    /*
    OVERRIDE doesn't work, the cpp side only call the cpp method and not the python one.
    https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
    */
    class PyCustomTransformation : public CustomTransformation {
        // Trampoline class
    public:
        using CustomTransformation::CustomTransformation;

        void apply_int(
        Tensor<int>& frame,
        Tensor<int>& labels,
        std::vector<std::shared_ptr<ROI> >& labelsROI,
        int id) override 
        {
            PYBIND11_OVERLOAD(
                void,                       /* Return type */
                CustomTransformation,       /* Parent class */
                apply_int,                  /* Name of function in C++ (must match Python name) */
                frame,                      /* Argument(s) */
                labels,
                labelsROI,
                id
            );
        }
        void apply_float(
            Tensor<float>& frame,
            Tensor<int>& labels,
            std::vector<std::shared_ptr<ROI> >& labelsROI,
            int id) override 
        {
            PYBIND11_OVERLOAD(
                void,                       /* Return type */
                CustomTransformation,       /* Parent class */
                apply_float,                /* Name of function in C++ (must match Python name) */
                frame,                      /* Argument(s) */
                labels,
                labelsROI,
                id
            );
#if SIZE_MAX != 0xFFFFFFFF
        }
        void apply_double(
            Tensor<double>& frame,
            Tensor<int>& labels,
            std::vector<std::shared_ptr<ROI> >& labelsROI,
            int id) override 
        {
            PYBIND11_OVERLOAD(
                void,                       /* Return type */
                CustomTransformation,       /* Parent class */
                apply_double,               /* Name of function in C++ (must match Python name) */
                frame,                      /* Argument(s) */
                labels,
                labelsROI,
                id
            );
#endif
        }
        void apply_unsigned_char(
            Tensor<unsigned char>& frame,
            Tensor<int>& labels,
            std::vector<std::shared_ptr<ROI> >& labelsROI,
            int id) override 
        {
            PYBIND11_OVERLOAD(
                void,                       /* Return type */
                CustomTransformation,       /* Parent class */
                apply_unsigned_char,        /* Name of function in C++ (must match Python name) */
                frame,                      /* Argument(s) */
                labels,
                labelsROI,
                id
            );
            
        }
        void apply_char(
            Tensor<char>& frame,
            Tensor<int>& labels,
            std::vector<std::shared_ptr<ROI> >& labelsROI,
            int id) override 
        {
            PYBIND11_OVERLOAD(
                void,                       /* Return type */
                CustomTransformation,       /* Parent class */
                apply_char,                 /* Name of function in C++ (must match Python name) */
                frame,                      /* Argument(s) */
                labels,
                labelsROI,
                id
            );
        }
        void apply_short(
            Tensor<short>& frame,
            Tensor<int>& labels,
            std::vector<std::shared_ptr<ROI> >& labelsROI,
            int id) override 
        {
            PYBIND11_OVERLOAD(
                void,                       /* Return type */
                CustomTransformation,       /* Parent class */
                apply_short,                /* Name of function in C++ (must match Python name) */
                frame,                      /* Argument(s) */
                labels,
                labelsROI,
                id
            );
        }
        void apply_unsigned_short(
            Tensor<unsigned short>& frame,
            Tensor<int>& labels,
            std::vector<std::shared_ptr<ROI> >& labelsROI,
            int id) override 
        {
            PYBIND11_OVERLOAD(
                void,                       /* Return type */
                CustomTransformation,       /* Parent class */
                apply_unsigned_short,       /* Name of function in C++ (must match Python name) */
                frame,                      /* Argument(s) */
                labels,
                labelsROI,
                id
            );
        }
    };
#if SIZE_MAX != 0xFFFFFFFF
        

    void init_CustomTransformation(py::module &m) {
        py::class_<CustomTransformation, std::shared_ptr<CustomTransformation>, PyCustomTransformation, Transformation>(m, "CustomTransformation", py::multiple_inheritance())
        .def(py::init<>())
        .def("apply_int", &CustomTransformation::apply_int, py::arg("frame"), py::arg("labels"), py::arg("labelsROI"), py::arg("id"))
        .def("apply_float", &CustomTransformation::apply_float, py::arg("frame"), py::arg("labels"), py::arg("labelsROI"), py::arg("id"))
        .def("apply_double", &CustomTransformation::apply_double, py::arg("frame"), py::arg("labels"), py::arg("labelsROI"), py::arg("id"))
        .def("apply_unsigned_char", &CustomTransformation::apply_unsigned_char, py::arg("frame"), py::arg("labels"), py::arg("labelsROI"), py::arg("id"))
        .def("apply_char", &CustomTransformation::apply_char, py::arg("frame"), py::arg("labels"), py::arg("labelsROI"), py::arg("id"))
        .def("apply_unsigned_short", &CustomTransformation::apply_unsigned_short, py::arg("frame"), py::arg("labels"), py::arg("labelsROI"), py::arg("id"))
        .def("apply_short", &CustomTransformation::apply_short, py::arg("frame"), py::arg("labels"), py::arg("labelsROI"), py::arg("id"))
        ;   
    }
#endif
}
