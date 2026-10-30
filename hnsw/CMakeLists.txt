cmake_minimum_required(VERSION 3.15)
project(knnosam CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- MACOS (Homebrew) ----------------
if(APPLE)
  # Homebrew prefixes
  execute_process(COMMAND brew --prefix OUTPUT_VARIABLE HOMEBREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(FAISS_ROOT   "${HOMEBREW_PREFIX}")
  execute_process(COMMAND brew --prefix libomp    OUTPUT_VARIABLE LIBOMP_ROOT    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND brew --prefix openblas  OUTPUT_VARIABLE OPENBLAS_ROOT  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Includes & libs
  include_directories(${FAISS_ROOT}/include ${LIBOMP_ROOT}/include ${OPENBLAS_ROOT}/include)
  link_directories(${FAISS_ROOT}/lib ${LIBOMP_ROOT}/lib ${OPENBLAS_ROOT}/lib)

  # executables
  add_executable(extract_hnsw extract_hnsw.cpp)
  add_executable(search_hnsw  search_hnsw.cpp)

  # Link (both targets!)
  target_link_libraries(extract_hnsw PRIVATE faiss omp openblas)
  target_link_libraries(search_hnsw  PRIVATE faiss omp openblas)

  # AppleClang OpenMP compile flag
  if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(extract_hnsw PRIVATE -Xpreprocessor -fopenmp)
    target_compile_options(search_hnsw  PRIVATE -Xpreprocessor -fopenmp)
  endif()

  # rpath so the binaries find dylibs at runtime
  set_target_properties(extract_hnsw search_hnsw PROPERTIES
    BUILD_RPATH   "${FAISS_ROOT}/lib;${LIBOMP_ROOT}/lib;${OPENBLAS_ROOT}/lib"
    INSTALL_RPATH "${FAISS_ROOT}/lib;${LIBOMP_ROOT}/lib;${OPENBLAS_ROOT}/lib"
  )

# ---------------- LINUX ----------------
else()
  include(CheckCXXCompilerFlag)

  # OpenMP (optional, but try to use it)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
  else()
    message(WARNING "OpenMP not found; building without it.")
  endif()

  # BLAS (OpenBLAS/MKL/ATLAS). Prefer OpenBLAS but any BLAS works.
  find_package(BLAS REQUIRED)
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")

  # FAISS: try proper CMake package first, then fallback to raw headers+lib
  set(_have_faiss FALSE)
  find_package(FAISS CONFIG QUIET)
  if (FAISS_FOUND)
    set(_have_faiss TRUE)
  else()
    find_path(FAISS_INCLUDE_DIR faiss/Index.h
      HINTS $ENV{FAISS_ROOT}/include /usr/include /usr/local/include)
    find_library(FAISS_LIBRARY NAMES faiss
      HINTS $ENV{FAISS_ROOT}/lib /usr/lib /usr/local/lib)
    if (FAISS_INCLUDE_DIR AND FAISS_LIBRARY)
      set(_have_faiss TRUE)
    endif()
  endif()

  if (NOT _have_faiss)
    message(FATAL_ERROR "FAISS not found. Install 'libfaiss-dev' (Debian/Ubuntu) or build from source, or set FAISS_ROOT.")
  endif()

  add_executable(extract_hnsw extract_hnsw.cpp)
  add_executable(search_hnsw  search_hnsw.cpp)

  # Include dirs if we didn't get an imported FAISS target
  if (FAISS_FOUND)
    # prefer imported target
    # target_link_libraries(extract_hnsw PRIVATE FAISS::faiss)
    # target_link_libraries(search_hnsw  PRIVATE FAISS::faiss)
    target_link_libraries(extract_hnsw PRIVATE faiss)
    target_link_libraries(search_hnsw  PRIVATE faiss)
  else()
    target_include_directories(extract_hnsw PRIVATE ${FAISS_INCLUDE_DIR})
    target_include_directories(search_hnsw  PRIVATE ${FAISS_INCLUDE_DIR})
    target_link_libraries(extract_hnsw PRIVATE ${FAISS_LIBRARY})
    target_link_libraries(search_hnsw  PRIVATE ${FAISS_LIBRARY})
  endif()

  # BLAS
  target_link_libraries(extract_hnsw PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(search_hnsw  PRIVATE ${BLAS_LIBRARIES})

  # OpenMP if available
  if (OpenMP_CXX_FOUND)
    target_link_libraries(extract_hnsw PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(search_hnsw  PRIVATE OpenMP::OpenMP_CXX)
  endif()

  # RPATH so binaries can relocate with siblings
  set_target_properties(extract_hnsw search_hnsw PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
  )
endif()
