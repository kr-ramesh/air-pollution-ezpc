cmake_minimum_required (VERSION 3.0)
project (MY_PROJ)
set(CMAKE_BUILD_TYPE Debug)
find_package(SCI REQUIRED PATHS "/home/t-anweshb/Desktop/Secfloat/SCI/build/install")

add_executable(gcnlstm-beacon gcnlstm_run.cpp)
target_link_libraries(gcnlstm-beacon SCI::SCI-FloatBeacon)
target_compile_options(gcnlstm-beacon PRIVATE "-w")

add_executable(gcnlstm-floatml gcnlstm_run.cpp)
target_link_libraries(gcnlstm-floatml SCI::SCI-FloatML)
target_compile_options(gcnlstm-floatml PRIVATE "-w") 

add_executable(gcnlstm-fixed gcnlstm_fixed.cpp)
target_link_libraries(gcnlstm-fixed SCI::SCI-FloatBeacon)
target_compile_options(gcnlstm-fixed PRIVATE "-w")
