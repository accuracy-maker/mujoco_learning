cmake_minimum_required(VERSION 3.16)
project(overview_simple)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pinocchio REQUIRED)

add_executable(overview-simple overview-simple.cpp)

target_link_libraries(overview-simple PRIVATE pinocchio::pinocchio)