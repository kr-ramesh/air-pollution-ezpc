cmake_minimum_required (VERSION 3.0)
project (MY_PROJ)
set(CMAKE_BUILD_TYPE Debug)
find_package(SCI REQUIRED PATHS "/home/t-anweshb/Desktop/Beacon/SCI/build/install")

add_executable(gcnlstm_clear gcnlstm_training_clear.cpp)
target_link_libraries(gcnlstm_clear SCI::SCI-FloatML)
target_compile_options(gcnlstm_clear PRIVATE "-w")

#add_executable(gcnlstm-beacon gcnlstm_run.cpp)
#target_link_libraries(gcnlstm-beacon SCI::SCI-FloatBeacon)
#target_compile_options(gcnlstm-beacon PRIVATE "-w")
#
#add_executable(gcnlstm-floatml gcn_split.cpp)
#target_link_libraries(gcnlstm-floatml SCI::SCI-FloatML)
#target_compile_options(gcnlstm-floatml PRIVATE "-w") 
#
#add_executable(gcnlstm-nocin-beacon gcnlstm_nocin.cpp)
#target_link_libraries(gcnlstm-nocin-beacon SCI::SCI-FloatBeacon)
#target_compile_options(gcnlstm-nocin-beacon PRIVATE "-w")
#
#add_executable(gcnlstm-nocin-floatml gcnlstm_nocin.cpp)
#target_link_libraries(gcnlstm-nocin-floatml SCI::SCI-FloatML)
#target_compile_options(gcnlstm-nocin-floatml PRIVATE "-w")
#
#add_executable(gcnlstm-cin-beacon gcnlstm_cin.cpp)
#target_link_libraries(gcnlstm-cin-beacon SCI::SCI-FloatBeacon)
#target_compile_options(gcnlstm-cin-beacon PRIVATE "-w")
#
#add_executable(gcnlstm-cin-floatml gcnlstm_cin.cpp)
#target_link_libraries(gcnlstm-cin-floatml SCI::SCI-FloatML)
#target_compile_options(gcnlstm-cin-floatml PRIVATE "-w") 
#
#add_executable(gcnlstm-fixed-beacon gcnlstm_fixed_adj.cpp)
#target_link_libraries(gcnlstm-fixed-beacon SCI::SCI-FloatML)
#target_compile_options(gcnlstm-fixed-beacon PRIVATE "-w")
#
#add_executable(gcnlstm-fixed-floatml gcnlstm_fixed_variable.cpp)
#target_link_libraries(gcnlstm-fixed-floatml SCI::SCI-FloatBeacon)
#target_compile_options(gcnlstm-fixed-floatml PRIVATE "-w")
