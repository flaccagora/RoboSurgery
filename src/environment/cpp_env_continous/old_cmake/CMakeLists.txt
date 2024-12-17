cmake_minimum_required(VERSION 3.12)
project(gridworld)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

pybind11_add_module(gridworld env.cpp)
