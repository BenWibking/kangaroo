cmake_minimum_required(VERSION 3.22)

project(kangaroo_runtime LANGUAGES C CXX)

option(KANGAROO_ASAN "Enable AddressSanitizer for kangaroo targets" OFF)
option(KANGAROO_USE_OPENPMD "Enable openPMD reader support" OFF)
option(KANGAROO_USE_PARTHENON_HDF5 "Enable Parthenon HDF5 reader support" ON)
option(KANGAROO_BUILD_CLAY_DASHBOARD "Build native Clay dashboard app" ON)

# msgpack-cxx's CMake config can still rely on FindBoost; keep legacy behavior
# under CMake 4.x to avoid noisy developer warnings.
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

find_package(HPX REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED)
find_package(amrexpr CONFIG QUIET)
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

if(NOT amrexpr_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    amrexpr
    GIT_REPOSITORY https://github.com/AMReX-Codes/amrex-parser.git
    GIT_TAG 9af95c252c25c95371f655d40d3a4bfb1de2dab3
  )
  FetchContent_MakeAvailable(amrexpr)
endif()

set(KANGAROO_RUNTIME_SOURCES
  src/adjacency.cpp
  src/backend_parthenon.cpp
  src/backend_memory.cpp
  src/backend_plotfile.cpp
  src/data_service_local.cpp
  src/executor.cpp
  src/kernel_registry.cpp
  src/plan_decode_msgpack.cpp
  src/runtime.cpp
)

if(KANGAROO_USE_OPENPMD)
  list(APPEND KANGAROO_RUNTIME_SOURCES src/backend_openpmd.cpp)
endif()

add_library(kangaroo_runtime STATIC ${KANGAROO_RUNTIME_SOURCES})

target_include_directories(kangaroo_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kangaroo_runtime PUBLIC HPX::hpx)
if(TARGET amrexpr::amrexpr)
  target_link_libraries(kangaroo_runtime PUBLIC amrexpr::amrexpr)
elseif(TARGET amrexpr)
  target_link_libraries(kangaroo_runtime PUBLIC amrexpr)
else()
  message(FATAL_ERROR "amrexpr target not available after FindPackage/FetchContent.")
endif()

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

if(KANGAROO_BUILD_CLAY_DASHBOARD)
  include(FetchContent)

  function(kangaroo_suppress_deprecation_warnings target_name)
    if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
      if(TARGET "${target_name}")
        target_compile_options("${target_name}" PRIVATE "-Wno-deprecated-declarations")
      endif()
    endif()
  endfunction()

  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)

  FetchContent_Declare(
    SDL
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.2.4
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(SDL)
  set_property(DIRECTORY "${sdl_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  kangaroo_suppress_deprecation_warnings(SDL3-shared)
  kangaroo_suppress_deprecation_warnings(SDL3-static)
  kangaroo_suppress_deprecation_warnings(SDL_uclibc)

  FetchContent_Declare(
    SDL_ttf
    GIT_REPOSITORY https://github.com/libsdl-org/SDL_ttf.git
    GIT_TAG release-3.2.2
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(SDL_ttf)
  set_property(DIRECTORY "${sdl_ttf_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  kangaroo_suppress_deprecation_warnings(SDL3_ttf-shared)
  kangaroo_suppress_deprecation_warnings(SDL3_ttf-static)

  FetchContent_Declare(
    SDL_image
    GIT_REPOSITORY https://github.com/libsdl-org/SDL_image.git
    GIT_TAG release-3.2.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(SDL_image)
  set_property(DIRECTORY "${SDL_image_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  kangaroo_suppress_deprecation_warnings(SDL3_image-shared)
  kangaroo_suppress_deprecation_warnings(SDL3_image-static)

  add_executable(kangaroo_dashboard_clay src/kangaroo_dashboard_clay.cpp)
  target_compile_features(kangaroo_dashboard_clay PRIVATE cxx_std_20)
  target_link_libraries(kangaroo_dashboard_clay PRIVATE
    nlohmann_json::nlohmann_json
    SDL3::SDL3
    SDL3_ttf::SDL3_ttf
    SDL3_image::SDL3_image
  )
endif()
