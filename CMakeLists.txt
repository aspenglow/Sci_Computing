cmake_minimum_required (VERSION 3.0)
project (PCSC-project)

set(CMAKE_CXX_STANDARD 11)

add_library(pcsc
  src/pcsc.cc
)

#add_subdirectory(googletest)

add_executable(main src/main.cc)
find_package( OpenCV 3 REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main pcsc)

add_executable(test_pcsc test/test.cc)
target_link_libraries(test_pcsc ${OpenCV_LIBS})
target_link_libraries(test_pcsc gtest_main gtest pthread pcsc)

#add_custom_target(test ./test_pcsc DEPENDS test_pcsc)
