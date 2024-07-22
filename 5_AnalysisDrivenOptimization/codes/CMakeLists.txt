cmake_minimum_required(VERSION 3.18)

# set the project name
project(ado
	VERSION 1.0
        LANGUAGES CXX)

# specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(adoConfig.h.in adoConfig.h)

# add the executable
add_executable(ado main.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    target_link_libraries(ado PUBLIC OpenMP::OpenMP_CXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O2")
endif()

# add the include library
add_subdirectory("utilities/src")
add_subdirectory("solvers/src")

target_link_libraries(ado PUBLIC UtilitiesCMake)
target_link_libraries(ado PUBLIC SolversCMake)

message("CXX compiler:")
message(${CMAKE_CXX_COMPILER})
message("CXX flags:")
message(${CMAKE_CXX_FLAGS})

target_include_directories(ado PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_BINARY_DIR}/utilities/src"
                           "${PROJECT_BINARY_DIR}/solvers/src"
                           )
