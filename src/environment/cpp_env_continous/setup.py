from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "gridworld",
        ["env.cpp"],  # Replace with your file name
        cxx_std=17,
    ),
]

setup(
    name="gridworld",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
