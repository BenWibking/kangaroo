cmake_minimum_required(VERSION 3.22)

project(kangaroo_runtime LANGUAGES CXX)

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

add_library(kangaroo_runtime STATIC
  src/adjacency.cpp
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

if(NOT msgpack-cxx_FOUND AND msgpack_SOURCE_DIR)
  target_include_directories(kangaroo_runtime PUBLIC "${msgpack_SOURCE_DIR}/include")
endif()

target_compile_features(kangaroo_runtime PUBLIC cxx_std_20)

target_compile_definitions(kangaroo_runtime PUBLIC KANGAROO_USE_MSGPACK)

nanobind_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE kangaroo_runtime)
target_compile_definitions(_core PRIVATE KANGAROO_USE_NANOBIND)

if(DEFINED SKBUILD_PLATLIB_DIR)
  set(KANGAROO_PYTHON_SITE "${SKBUILD_PLATLIB_DIR}")
else()
  set(KANGAROO_PYTHON_SITE
      "${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
endif()
set(KANGAROO_PYTHON_SITE
    "${KANGAROO_PYTHON_SITE}"
    CACHE PATH "Python site-packages directory for install")

install(TARGETS _core LIBRARY DESTINATION "${KANGAROO_PYTHON_SITE}/analysis")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../analysis/"
        DESTINATION "${KANGAROO_PYTHON_SITE}/analysis"
        FILES_MATCHING PATTERN "*.py")
