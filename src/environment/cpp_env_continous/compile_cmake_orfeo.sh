ml load cmake

# Create a build directory
mkdir build && cd build

# Configure the build
pybind11_dir=$(python -m pybind11 --cmakedir)
echo $pibind11_dir
export LD_LIBRARY_PATH=/usr/lib64/libstdc++.so.6:$LD_LIBRARY_PATH

cmake -Dpybind11_DIR=$pybind11_dir ..

# Build the module
make
