cmake_minimum_required (VERSION 3.0)
project (MY_PROJ)
find_package(SCI REQUIRED PATHS "/home/t-anweshb/Desktop/Secfloat/SCI/build/install")

add_executable(float_to_int float_to_int.cpp)
target_link_libraries(float_to_int SCI::SCI-FloatingPoint)
