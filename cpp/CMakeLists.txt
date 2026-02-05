cmake_minimum_required(VERSION 3.22)

project(kangaroo_runtime LANGUAGES CXX)

option(KANGAROO_ASAN "Enable AddressSanitizer for kangaroo targets" OFF)
option(KANGAROO_USE_OPENPMD "Enable openPMD reader support" ON)
option(KANGAROO_USE_PARTHENON_HDF5 "Enable Parthenon HDF5 reader support" ON)

find_package(HPX REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED)
find_package(msgpack-cxx CONFIG QUIET)
if(NOT msgpack-cxx_FOUND)
  # CMake 4.x requires a policy minimum for older subprojects like msgpack.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  include(FetchContent)
  FetchContent_Declare(
    msgpack
    GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
    GIT_TAG cpp-6.1.0
  )
  FetchContent_MakeAvailable(msgpack)
  if(NOT TARGET msgpackc-cxx AND TARGET msgpack-cxx)
    add_library(msgpackc-cxx ALIAS msgpack-cxx)
  endif()
endif()

# If HPX is using tcmalloc, try to link it into standalone modules too
# to keep allocator ownership consistent across extensions.
find_library(TCMALLOC_MINIMAL_LIB
  NAMES tcmalloc_minimal
  HINTS
    ${CMAKE_PREFIX_PATH}/lib
    $ENV{CONDA_PREFIX}/lib
)

add_library(kangaroo_runtime STATIC
  src/adjacency.cpp
  src/backend_parthenon.cpp
  src/backend_memory.cpp
  src/backend_plotfile.cpp
  src/backend_openpmd.cpp
  src/data_service_local.cpp
  src/executor.cpp
  src/kernel_registry.cpp
  src/plan_decode_msgpack.cpp
  src/runtime.cpp
)

target_include_directories(kangaroo_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kangaroo_runtime PUBLIC HPX::hpx)

if(KANGAROO_USE_PARTHENON_HDF5)
  find_package(HDF5 COMPONENTS C QUIET)
  if(HDF5_FOUND)
    target_compile_definitions(kangaroo_runtime PUBLIC KANGAROO_USE_PARTHENON_HDF5)
    target_include_directories(kangaroo_runtime PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kangaroo_runtime PUBLIC ${HDF5_LIBRARIES})
  else()
    message(WARNING "HDF5 C library not found; disabling Parthenon HDF5 backend.")
  endif()
endif()

if(KANGAROO_USE_OPENPMD)
  find_package(openPMD CONFIG QUIET)
  if(NOT openPMD_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      openPMD
      GIT_REPOSITORY https://github.com/openPMD/openPMD-api.git
      GIT_TAG 0.17.0
    )
    FetchContent_MakeAvailable(openPMD)
  endif()
  if(NOT TARGET openPMD::openPMD)
    message(FATAL_ERROR "openPMD target not available after FetchContent/FindPackage.")
  endif()
  target_compile_definitions(kangaroo_runtime PUBLIC KANGAROO_USE_OPENPMD)
  target_link_libraries(kangaroo_runtime PUBLIC openPMD::openPMD)
endif()

if(NOT msgpack-cxx_FOUND AND msgpack_SOURCE_DIR)
  target_include_directories(kangaroo_runtime PUBLIC "${msgpack_SOURCE_DIR}/include")
endif()

target_compile_features(kangaroo_runtime PUBLIC cxx_std_20)

target_compile_definitions(kangaroo_runtime PUBLIC KANGAROO_USE_MSGPACK)

nanobind_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE kangaroo_runtime)
target_compile_definitions(_core PRIVATE KANGAROO_USE_NANOBIND)
if(TCMALLOC_MINIMAL_LIB)
  target_link_libraries(_core PRIVATE ${TCMALLOC_MINIMAL_LIB})
endif()

add_library(kangaroo_plotfile STATIC
  src/plotfile_reader.cpp
)

target_include_directories(kangaroo_plotfile PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(kangaroo_plotfile PUBLIC cxx_std_20)

target_link_libraries(kangaroo_runtime PUBLIC kangaroo_plotfile)

nanobind_add_module(_plotfile src/plotfile_bindings.cpp)
target_link_libraries(_plotfile PRIVATE kangaroo_plotfile)
target_compile_definitions(_plotfile PRIVATE KANGAROO_USE_NANOBIND)
if(TCMALLOC_MINIMAL_LIB)
  target_link_libraries(kangaroo_plotfile PUBLIC ${TCMALLOC_MINIMAL_LIB})
  target_link_libraries(_plotfile PRIVATE ${TCMALLOC_MINIMAL_LIB})
endif()

if(KANGAROO_ASAN)
  set(_asan_flags "-fsanitize=address" "-fno-omit-frame-pointer")
  foreach(target kangaroo_runtime _core kangaroo_plotfile _plotfile)
    target_compile_options(${target} PRIVATE ${_asan_flags})
    target_link_options(${target} PRIVATE "-fsanitize=address")
  endforeach()
endif()

if(DEFINED SKBUILD_PLATLIB_DIR)
  set(KANGAROO_PYTHON_SITE "${SKBUILD_PLATLIB_DIR}")
else()
  set(KANGAROO_PYTHON_SITE
      "${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
endif()
set(KANGAROO_PYTHON_SITE
    "${KANGAROO_PYTHON_SITE}"
    CACHE PATH "Python site-packages directory for install")

install(TARGETS _core _plotfile LIBRARY DESTINATION "${KANGAROO_PYTHON_SITE}/analysis")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../analysis/"
        DESTINATION "${KANGAROO_PYTHON_SITE}/analysis"
        FILES_MATCHING PATTERN "*.py")
