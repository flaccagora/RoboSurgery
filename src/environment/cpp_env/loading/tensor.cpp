
#include <torch/script.h>

#include <iostream>
#include <memory>


//  mkdir build       
// cd build
// cmake -DCMAKE_PREFIX_PATH=/usr/local/include/libtorch ..
// cmake --build . --config Release

int save_tensor() {
  auto x = torch::ones({3, 3,3});
  auto bytes = torch::jit::pickle_save(x);
  std::ofstream fout("x.zip", std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
  
  std::cout << x << std::endl;

  return 0;
}

void iterate(torch::Tensor x) {
  // iterate over tensor

  // assert foo is 2-dimensional and holds floats.
  auto foo_a = x.accessor<float,3>();
  float trace = 0;

  for(int i = 0; i < foo_a.size(0); i++) {
    for (int j = 0; j < foo_a.size(1); j++) {
      for (int k = 0; k < foo_a.size(2); k++) {
        foo_a[i][j][k] = i*foo_a.size(1) + j + k;
      }
    }
  }

  std::cout << x << std::endl;
}

int main() {
  save_tensor();
  return 0;
}