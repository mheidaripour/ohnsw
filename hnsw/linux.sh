sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev libomp-dev libfaiss-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

