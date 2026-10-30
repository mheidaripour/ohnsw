brew install cmake faiss openblas libomp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

