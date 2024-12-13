rm -rf build
conda activate robogym

# Create a build directory
mkdir build && cd build

# Configure the build
pybind11_dir=$(python -m pybind11 --cmakedir)

cmake -Dpybind11_DIR=$pybind11_dir ..

# Build the module
make
