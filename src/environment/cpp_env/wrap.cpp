// env_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include "env.h"

namespace py = pybind11;

PYBIND11_MODULE(wrap, m) {
    py::class_<GridEnvDeform>(m, "GridEnvDeform")
        .def(py::init<const Eigen::MatrixXi&, int, int, int, int>());
        // .def("reset", &GridEnvDeform::reset)
        // .def("step", &GridEnvDeform::step)
        // .def("render", &GridEnvDeform::render);
}
