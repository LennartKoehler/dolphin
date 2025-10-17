# Diff Details

Date : 2025-10-14 15:31:32

Directory /home/lennart-k-hler/projects/dolphin

Total : 87 files,  516 codes, -190 comments, 156 blanks, all 482 lines

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [CMakeLists.txt](/CMakeLists.txt) | CMake | 6 | 0 | 0 | 6 |
| [README\_dev.md](/README_dev.md) | Markdown | 8 | 0 | 0 | 8 |
| [TODO.md](/TODO.md) | Markdown | -8 | 0 | -10 | -18 |
| [backends/cpu/CMakeLists.txt](/backends/cpu/CMakeLists.txt) | CMake | 1 | 0 | 0 | 1 |
| [backends/cpu/CPUBackend.cpp](/backends/cpu/CPUBackend.cpp) | C++ | -45 | -170 | -22 | -237 |
| [backends/cpu/CPUBackend.h](/backends/cpu/CPUBackend.h) | C++ | 3 | -8 | 0 | -5 |
| [backends/cuda/CMakeLists.txt](/backends/cuda/CMakeLists.txt) | CMake | 1 | 0 | 1 | 2 |
| [backends/cuda/CUDABackend.cpp](/backends/cuda/CUDABackend.cpp) | C++ | -97 | -21 | 13 | -105 |
| [backends/cuda/CUDABackend.h](/backends/cuda/CUDABackend.h) | C++ | 2 | -1 | 2 | 3 |
| [build/.cmake/api/v1/reply/cache-v2-d5676415438519c025b0.json](/build/.cmake/api/v1/reply/cache-v2-d5676415438519c025b0.json) | JSON | 3,839 | 0 | 1 | 3,840 |
| [build/.cmake/api/v1/reply/cache-v2-f4fa587d059c9e4d5e7a.json](/build/.cmake/api/v1/reply/cache-v2-f4fa587d059c9e4d5e7a.json) | JSON | -3,839 | 0 | -1 | -3,840 |
| [build/.cmake/api/v1/reply/cmakeFiles-v1-2506b16d0ea5eeff6166.json](/build/.cmake/api/v1/reply/cmakeFiles-v1-2506b16d0ea5eeff6166.json) | JSON | 444 | 0 | 1 | 445 |
| [build/.cmake/api/v1/reply/cmakeFiles-v1-4cc205a9fb6c1a8ec484.json](/build/.cmake/api/v1/reply/cmakeFiles-v1-4cc205a9fb6c1a8ec484.json) | JSON | -429 | 0 | -1 | -430 |
| [build/.cmake/api/v1/reply/codemodel-v2-04bf2255868ecc90e74a.json](/build/.cmake/api/v1/reply/codemodel-v2-04bf2255868ecc90e74a.json) | JSON | 202 | 0 | 1 | 203 |
| [build/.cmake/api/v1/reply/codemodel-v2-ce6666eb84419e2312d2.json](/build/.cmake/api/v1/reply/codemodel-v2-ce6666eb84419e2312d2.json) | JSON | -193 | 0 | -1 | -194 |
| [build/.cmake/api/v1/reply/index-2025-10-02T07-32-33-0498.json](/build/.cmake/api/v1/reply/index-2025-10-02T07-32-33-0498.json) | JSON | -132 | 0 | -1 | -133 |
| [build/.cmake/api/v1/reply/index-2025-10-14T12-11-54-0683.json](/build/.cmake/api/v1/reply/index-2025-10-14T12-11-54-0683.json) | JSON | 132 | 0 | 1 | 133 |
| [build/.cmake/api/v1/reply/target-CUBE-Debug-15e190072a85188360dd.json](/build/.cmake/api/v1/reply/target-CUBE-Debug-15e190072a85188360dd.json) | JSON | 402 | 0 | 1 | 403 |
| [build/.cmake/api/v1/reply/target-CUBE-Debug-176658ca7faeaa4c739c.json](/build/.cmake/api/v1/reply/target-CUBE-Debug-176658ca7faeaa4c739c.json) | JSON | -402 | 0 | -1 | -403 |
| [build/.cmake/api/v1/reply/target-backendlib-Debug-6566de112071780eb842.json](/build/.cmake/api/v1/reply/target-backendlib-Debug-6566de112071780eb842.json) | JSON | 117 | 0 | 1 | 118 |
| [build/.cmake/api/v1/reply/target-cpu\_backend-Debug-55e5c256c481736ea780.json](/build/.cmake/api/v1/reply/target-cpu_backend-Debug-55e5c256c481736ea780.json) | JSON | -172 | 0 | -1 | -173 |
| [build/.cmake/api/v1/reply/target-cpu\_backend-Debug-d0bc37f3892c3ad99acd.json](/build/.cmake/api/v1/reply/target-cpu_backend-Debug-d0bc37f3892c3ad99acd.json) | JSON | 184 | 0 | 1 | 185 |
| [build/.cmake/api/v1/reply/target-cuda\_backend-Debug-63690b4a3685a64fb028.json](/build/.cmake/api/v1/reply/target-cuda_backend-Debug-63690b4a3685a64fb028.json) | JSON | -314 | 0 | -1 | -315 |
| [build/.cmake/api/v1/reply/target-cuda\_backend-Debug-e075bf1ee1a304c6c8a0.json](/build/.cmake/api/v1/reply/target-cuda_backend-Debug-e075bf1ee1a304c6c8a0.json) | JSON | 323 | 0 | 1 | 324 |
| [build/.cmake/api/v1/reply/target-dolphin-Debug-8081bc74c459b5b78bf5.json](/build/.cmake/api/v1/reply/target-dolphin-Debug-8081bc74c459b5b78bf5.json) | JSON | 401 | 0 | 1 | 402 |
| [build/.cmake/api/v1/reply/target-dolphin-Debug-f3534103293058f1a090.json](/build/.cmake/api/v1/reply/target-dolphin-Debug-f3534103293058f1a090.json) | JSON | -392 | 0 | -1 | -393 |
| [build/.cmake/api/v1/reply/target-dolphinlib-Debug-d8c1ae96fd03915a5e2b.json](/build/.cmake/api/v1/reply/target-dolphinlib-Debug-d8c1ae96fd03915a5e2b.json) | JSON | -556 | 0 | -1 | -557 |
| [build/.cmake/api/v1/reply/target-dolphinlib-Debug-df92e881f86710f6b214.json](/build/.cmake/api/v1/reply/target-dolphinlib-Debug-df92e881f86710f6b214.json) | JSON | 564 | 0 | 1 | 565 |
| [build/.cmake/api/v1/reply/target-imgui\_lib-Debug-10cc2d11926981f69cef.json](/build/.cmake/api/v1/reply/target-imgui_lib-Debug-10cc2d11926981f69cef.json) | JSON | -180 | 0 | -1 | -181 |
| [build/.cmake/api/v1/reply/target-imgui\_lib-Debug-2a225024450bc706d21a.json](/build/.cmake/api/v1/reply/target-imgui_lib-Debug-2a225024450bc706d21a.json) | JSON | 180 | 0 | 1 | 181 |
| [build/.cmake/api/v1/reply/target-implot3d\_lib-Debug-56e6fe81174efbd158d4.json](/build/.cmake/api/v1/reply/target-implot3d_lib-Debug-56e6fe81174efbd158d4.json) | JSON | -162 | 0 | -1 | -163 |
| [build/.cmake/api/v1/reply/target-implot3d\_lib-Debug-d17e06329df9cfd5a732.json](/build/.cmake/api/v1/reply/target-implot3d_lib-Debug-d17e06329df9cfd5a732.json) | JSON | 162 | 0 | 1 | 163 |
| [build/CMakeFiles/Makefile.cmake](/build/CMakeFiles/Makefile.cmake) | CMake | 1 | 0 | 0 | 1 |
| [build/CMakeFiles/backendlib.dir/DependInfo.cmake](/build/CMakeFiles/backendlib.dir/DependInfo.cmake) | CMake | 17 | 0 | 7 | 24 |
| [build/CMakeFiles/backendlib.dir/cmake\_clean.cmake](/build/CMakeFiles/backendlib.dir/cmake_clean.cmake) | CMake | 10 | 0 | 2 | 12 |
| [build/CMakeFiles/backendlib.dir/cmake\_clean\_target.cmake](/build/CMakeFiles/backendlib.dir/cmake_clean_target.cmake) | CMake | 3 | 0 | 1 | 4 |
| [build/CMakeFiles/backendlib.dir/compiler\_depend.ts](/build/CMakeFiles/backendlib.dir/compiler_depend.ts) | TypeScript | 2 | 0 | 1 | 3 |
| [build/CMakeFiles/dolphinlib.dir/DependInfo.cmake](/build/CMakeFiles/dolphinlib.dir/DependInfo.cmake) | CMake | 1 | 0 | 0 | 1 |
| [build/CMakeFiles/dolphinlib.dir/cmake\_clean.cmake](/build/CMakeFiles/dolphinlib.dir/cmake_clean.cmake) | CMake | 2 | 0 | 0 | 2 |
| [build/Makefile](/build/Makefile) | Makefile | 43 | 10 | 15 | 68 |
| [build/compile\_commands.json](/build/compile_commands.json) | JSON | 12 | 0 | 0 | 12 |
| [build/imgui.ini](/build/imgui.ini) | Ini | -15 | 0 | -6 | -21 |
| [configs/default\_config.json](/configs/default_config.json) | JSON | 1 | 0 | 0 | 1 |
| [examples/memory\_exception\_example.cpp](/examples/memory_exception_example.cpp) | C++ | 28 | 9 | 10 | 47 |
| [include/Dolphin.h](/include/Dolphin.h) | C++ | -1 | 0 | 0 | -1 |
| [include/RectangleShape.h](/include/RectangleShape.h) | C++ | -37 | 0 | -3 | -40 |
| [include/ThreadPool.h](/include/ThreadPool.h) | C++ | -2 | -1 | 0 | -3 |
| [include/backend/BackendFactory.h](/include/backend/BackendFactory.h) | C++ | 46 | 2 | 13 | 61 |
| [include/backend/ComplexData.h](/include/backend/ComplexData.h) | C++ | 80 | 3 | 11 | 94 |
| [include/backend/Exceptions.h](/include/backend/Exceptions.h) | C++ | 63 | 4 | 15 | 82 |
| [include/backend/IBackendMemoryManager.h](/include/backend/IBackendMemoryManager.h) | C++ | 42 | 4 | 12 | 58 |
| [include/backend/IDeconvolutionBackend.h](/include/backend/IDeconvolutionBackend.h) | C++ | 81 | 10 | 35 | 126 |
| [include/complexType.h](/include/complexType.h) | C++ | -2 | 0 | -2 | -4 |
| [include/deconvolution/DeconvolutionBackendThreadManager.h](/include/deconvolution/DeconvolutionBackendThreadManager.h) | C++ | -15 | 0 | -6 | -21 |
| [include/deconvolution/DeconvolutionConfig.h](/include/deconvolution/DeconvolutionConfig.h) | C++ | 13 | -12 | 4 | 5 |
| [include/deconvolution/DeconvolutionProcessor.h](/include/deconvolution/DeconvolutionProcessor.h) | C++ | 5 | -15 | 0 | -10 |
| [include/deconvolution/IDeconvolutionBackend.h](/include/deconvolution/IDeconvolutionBackend.h) | C++ | -122 | -11 | -38 | -171 |
| [include/deconvolution/algorithms/DeconvolutionAlgorithm.h](/include/deconvolution/algorithms/DeconvolutionAlgorithm.h) | C++ | 7 | 0 | 1 | 8 |
| [include/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h](/include/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h) | C++ | 1 | -1 | 0 | 0 |
| [include/deconvolution/algorithms/RLADDeconvolutionAlgorithm.h](/include/deconvolution/algorithms/RLADDeconvolutionAlgorithm.h) | C++ | 1 | 0 | 0 | 1 |
| [include/deconvolution/algorithms/RLDeconvolutionAlgorithm.h](/include/deconvolution/algorithms/RLDeconvolutionAlgorithm.h) | C++ | 1 | 0 | -3 | -2 |
| [include/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h](/include/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h) | C++ | 1 | -1 | 0 | 0 |
| [include/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h](/include/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h) | C++ | 1 | 0 | 0 | 1 |
| [include/deconvolution/algorithms/TestAlgorithm.h](/include/deconvolution/algorithms/TestAlgorithm.h) | C++ | 2 | 0 | 1 | 3 |
| [legacy/DeconvolutionBackendThreadManager.cpp](/legacy/DeconvolutionBackendThreadManager.cpp) | C++ | 57 | 3 | 22 | 82 |
| [legacy/DeconvolutionBackendThreadManager.h](/legacy/DeconvolutionBackendThreadManager.h) | C++ | 22 | 0 | 6 | 28 |
| [legacy/unused/BaseDeconvolutionAlgorithmGPU.cpp](/legacy/unused/BaseDeconvolutionAlgorithmGPU.cpp) | C++ | 21 | 1 | 2 | 24 |
| [lib/cube/CMakeLists.txt](/lib/cube/CMakeLists.txt) | CMake | 1 | 0 | -1 | 0 |
| [lib/cube/include/operations.h](/lib/cube/include/operations.h) | C++ | 1 | 0 | 0 | 1 |
| [lib/cube/src/main.cpp](/lib/cube/src/main.cpp) | C++ | 9 | 9 | 0 | 18 |
| [lib/cube/src/operations.cu](/lib/cube/src/operations.cu) | CUDA C++ | 68 | 0 | 54 | 122 |
| [src/DeconvolutionService.cpp](/src/DeconvolutionService.cpp) | C++ | -1 | 0 | 0 | -1 |
| [src/PSFManager.cpp](/src/PSFManager.cpp) | C++ | 4 | 0 | 0 | 4 |
| [src/ThreadPool.cpp](/src/ThreadPool.cpp) | C++ | 2 | 0 | 0 | 2 |
| [src/backend/BackendFactory.cpp](/src/backend/BackendFactory.cpp) | C++ | 38 | 0 | 9 | 47 |
| [src/backend/ComplexData.cpp](/src/backend/ComplexData.cpp) | C++ | 32 | 3 | 8 | 43 |
| [src/deconvolution/DeconvolutionBackendThreadManager.cpp](/src/deconvolution/DeconvolutionBackendThreadManager.cpp) | C++ | -4 | 0 | -3 | -7 |
| [src/deconvolution/DeconvolutionConfig.cpp](/src/deconvolution/DeconvolutionConfig.cpp) | C++ | 0 | -13 | -2 | -15 |
| [src/deconvolution/DeconvolutionProcessor.cpp](/src/deconvolution/DeconvolutionProcessor.cpp) | C++ | -47 | 14 | 1 | -32 |
| [src/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.cpp](/src/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.cpp) | C++ | -2 | -1 | 2 | -1 |
| [src/deconvolution/algorithms/RLADDeconvolutionAlgorithm.cpp](/src/deconvolution/algorithms/RLADDeconvolutionAlgorithm.cpp) | C++ | 0 | 0 | 1 | 1 |
| [src/deconvolution/algorithms/RLDeconvolutionAlgorithm.cpp](/src/deconvolution/algorithms/RLDeconvolutionAlgorithm.cpp) | C++ | 2 | 1 | 0 | 3 |
| [src/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.cpp](/src/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.cpp) | C++ | -6 | -1 | 1 | -6 |
| [src/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.cpp](/src/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.cpp) | C++ | -6 | -1 | 0 | -7 |
| [src/deconvolution/algorithms/TestAlgorithm.cpp](/src/deconvolution/algorithms/TestAlgorithm.cpp) | C++ | 3 | 0 | 1 | 4 |
| [src/frontend/CLIFrontend.cpp](/src/frontend/CLIFrontend.cpp) | C++ | 1 | 0 | 0 | 1 |
| [src/frontend/gui/UIDeconvolutionConfig.cpp](/src/frontend/gui/UIDeconvolutionConfig.cpp) | C++ | 1 | -6 | 0 | -5 |

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details