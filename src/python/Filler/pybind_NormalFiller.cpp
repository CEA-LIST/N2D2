
#ifdef CUDA

#ifdef PYBIND
#include "Filler/NormalFiller.hpp"


#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace N2D2 {
template<typename T>
void declare_NormalFiller(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("NormalFiller_" + typeStr);

    py::class_<NormalFiller<T>, std::shared_ptr<NormalFiller<T>>, Filler> (
            m, pyClassName.c_str(), py::multiple_inheritance()
    )

    .def(py::init<double, double>(), py::arg("mean"), py::arg("stdDev"));
}

void init_NormalFiller(py::module &m) {
    declare_NormalFiller<float>(m, "float");
    declare_NormalFiller<double>(m, "double");
}
}


#endif

#endif
