# ISPC: Making CPU SIMD fun while tracing rays!
Masterclass by Pete Brubaker at Graphics Programming Conference 2024

https://www.graphicsprogrammingconference.nl/ispc-making-cpu-simd-fun-while-tracing-rays/
    
## Requirements
- CMake version 3.19 or higher
- ISPC for your OS 

## Build Instructions
Make and enter a build directory
```
mkdir build && cd build
```

### Windows
```
cmake -G "Ninja Multi-Config" --fresh ..
cmake --build . --config Release
```
