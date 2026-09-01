add_library(autograd STATIC
        ../include/autograd/value.h
        ../include/autograd/value.cpp)
target_include_directories(autograd PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(autograd PUBLIC cxx_std_17)