
#ifdef CUDA

#ifdef PYBIND
#include "Filler/HeFiller.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace N2D2 {
template<typename T>
void declare_HeFiller(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("HeFiller_" + typeStr);

    py::class_<HeFiller<T>, std::shared_ptr<HeFiller<T>>, Filler> filler(
            m, pyClassName.c_str(), py::multiple_inheritance()
    );

    py::enum_<class HeFiller<T>::VarianceNorm>(filler, "VarianceNorm")
    .value("FanIn", HeFiller<T>::VarianceNorm::FanIn)
    .value("Average", HeFiller<T>::VarianceNorm::Average)
    .value("FanOut", HeFiller<T>::VarianceNorm::FanOut)
    .export_values();

    filler
    .def(py::init<typename HeFiller<T>::VarianceNorm, T, T>(), 
        py::arg("varianceNorm") = HeFiller<T>::VarianceNorm::FanIn, 
        py::arg("meanNorm") = 0.0, 
        py::arg("scaling") = 1.0);
}

void init_HeFiller(py::module &m) {
    declare_HeFiller<float>(m, "float");
    declare_HeFiller<double>(m, "double");
}
}


#endif

#endif
