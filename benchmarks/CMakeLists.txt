add_executable(autograd_benchmarks
        main.cpp
)
target_link_libraries(autograd_benchmarks PRIVATE autograd)
target_include_directories(autograd_benchmarks PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(autograd_benchmarks PRIVATE cxx_std_17)
