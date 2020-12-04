
#ifdef CUDA

#ifdef PYBIND
#include "Filler/HeFiller.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;

/*
namespace N2D2 {
template<typename T>
void declare_HeFiller(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("HeFiller" + typeStr);

    py::class_<HeFiller<T>, std::shared_ptr<HeFiller<T>>, Filler> filler(
            m, pyClassName.c_str(), py::multiple_inheritance()
    );

    py::enum_<HeFiller<T>::VarianceNorm>(filler, "VarianceNorm")
    .value("FanIn", HeFiller<T>::VarianceNorm::FanIn)
    .value("Average", HeFiller<T>::VarianceNorm::Average)
    .value("FanOut", HeFiller<T>::VarianceNorm::FanOut)
    .export_values();

    filler

    .def(py::init<HeFiller<T>::VarianceNorm, T, T>(), 
        py::arg("varianceNorm") = HeFiller<T>::VarianceNorm::FanIn, 
        py::arg("meanNorm") = 0.0, 
        py::arg("scaling") = 1.0);
}

void init_HeFiller(py::module &m) {
    declare_HeFiller<float>(m, "float");
    declare_HeFiller<double>(m, "double");
}
}
*/

//TODO: Make this wwork with templated version of declare_HeFiller_float
namespace N2D2 {
void declare_HeFiller_float(py::module &m) {

    py::class_<HeFiller<float>, std::shared_ptr<HeFiller<float>>, Filler> filler(
            m, "HeFiller_float", py::multiple_inheritance()
    );

    py::enum_<HeFiller<float>::VarianceNorm>(filler, "VarianceNorm")
    .value("FanIn", HeFiller<float>::VarianceNorm::FanIn)
    .value("Average", HeFiller<float>::VarianceNorm::Average)
    .value("FanOut", HeFiller<float>::VarianceNorm::FanOut)
    .export_values();

    filler

    .def(py::init<HeFiller<float>::VarianceNorm, float, float>(), 
        py::arg("varianceNorm") = HeFiller<float>::VarianceNorm::FanIn, 
        py::arg("meanNorm") = 0.0, 
        py::arg("scaling") = 1.0);
}

void init_HeFiller(py::module &m) {
    declare_HeFiller_float(m);
}
}

#endif

#endif
