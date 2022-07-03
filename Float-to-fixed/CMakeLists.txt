cmake_minimum_required (VERSION 3.0)
project (MY_PROJ)
find_package(SCI REQUIRED PATHS "/home/anwesh/Secfloat/SCI/build/install")

add_executable(float_to_int float_to_int.cpp)
target_link_libraries(float_to_int SCI::SCI-FloatingPoint)

add_executable(fti_clear fti_clear.cpp)

add_executable(fti_sec fti_sec.cpp)
target_link_libraries(fti_sec SCI::SCI-FloatingPoint)
