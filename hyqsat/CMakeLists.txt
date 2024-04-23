cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(minisat)

#--------------------------------------------------------------------------------------------------
# Configurable options:
SET(CMAKE_CXX_FLAGS "-std=c++14")
option(STATIC_BINARIES "Link binaries statically." ON)
option(USE_SORELEASE   "Use SORELEASE in shared library filename." ON)

#--------------------------------------------------------------------------------------------------
# Library version:

set(MINISAT_SOMAJOR   2)
set(MINISAT_SOMINOR   1)
set(MINISAT_SORELEASE 0)

# Compute VERSION and SOVERSION:
if (USE_SORELEASE)
  set(MINISAT_VERSION ${MINISAT_SOMAJOR}.${MINISAT_SOMINOR}.${MINISAT_SORELEASE})
else()
  set(MINISAT_VERSION ${MINISAT_SOMAJOR}.${MINISAT_SOMINOR})
endif()
set(MINISAT_SOVERSION ${MINISAT_SOMAJOR})

#--------------------------------------------------------------------------------------------------
# Dependencies:

find_package(ZLIB)
include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${minisat_SOURCE_DIR})
#--------------------------------------------------------------------------------------------------
# python:
# 调试时，还是用usr/bin的更好一点
find_program(Python_EXECUTABLE python NO_CMAKE_FIND_ROOT_PATH)
find_package(Python REQUIRED COMPONENTS Development Interpreter)
include_directories(${PYTHON_INCLUDE_DIRS})
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message("Python3: " ${Python_INCLUDE_DIRS})
message("Python3: " ${Python_LIBRARIES})

#SET(Python3_INCLUDE_DIRS "/usr/include/python3.8/")
#SET(Python3_LIBRARIES "/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so")
#message("Python3: " ${Python3_INCLUDE_DIRS})
#message("Python3: " ${Python3_LIBRARIES})

include_directories(./include/ ${Python_INCLUDE_DIRS})

#--------------------------------------------------------------------------------------------------
# Compile flags:

add_definitions(-D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS)

#--------------------------------------------------------------------------------------------------
# Build Targets:

set(MINISAT_LIB_SOURCES
        minisat/utils/Options.cc
        minisat/utils/System.cc
        minisat/core/Solver.cc
        minisat/simp/SimpSolver.cc)

set(CMAKE_BUILD_TYPE )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g -ggdb")
#set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}/../lib")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})

add_library(minisat-lib-static STATIC ${MINISAT_LIB_SOURCES} )
add_library(minisat-lib-shared SHARED ${MINISAT_LIB_SOURCES} minisat/tabu/solve_sat_2.h  minisat/tabu/Simulate.h )
#minisat/core/innerSolver.cpp minisat/core/innerSolver.cpp.h
target_link_libraries(minisat-lib-shared ${ZLIB_LIBRARY})
target_link_libraries(minisat-lib-static ${ZLIB_LIBRARY})

#target_link_libraries(minisat-lib-shared -lpython3.6m)
#target_link_libraries(minisat-lib-static -lpython3.6m)
target_link_libraries(minisat-lib-shared ${Python_LIBRARIES})
target_link_libraries(minisat-lib-static ${Python_LIBRARIES})

target_link_libraries(minisat-lib-shared pthread)
target_link_libraries(minisat-lib-static pthread)

add_executable(minisat_core minisat/core/Main.cc minisat/core/Solver.h)
add_executable(minisat_simp minisat/simp/Main.cc)

if(STATIC_BINARIES)
  target_link_libraries(minisat_core minisat-lib-static)
  target_link_libraries(minisat_simp minisat-lib-static)
else()
  target_link_libraries(minisat_core minisat-lib-shared)
  target_link_libraries(minisat_simp minisat-lib-shared)
endif()

set_target_properties(minisat-lib-static PROPERTIES OUTPUT_NAME "minisat_core")
set_target_properties(minisat-lib-shared
        PROPERTIES
        OUTPUT_NAME "minisat"
        VERSION ${MINISAT_VERSION}
        SOVERSION ${MINISAT_SOVERSION})

set_target_properties(minisat_core       PROPERTIES OUTPUT_NAME "minisat_core")

#--------------------------------------------------------------------------------------------------
# Installation targets:

install(TARGETS minisat-lib-static minisat-lib-shared minisat_core minisat_simp
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION static_lib)

install(DIRECTORY minisat/mtl minisat/utils minisat/core minisat/tabu minisat/simp
        DESTINATION include/minisat
        FILES_MATCHING PATTERN "*.h")
