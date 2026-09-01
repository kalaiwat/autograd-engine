include(FetchContent)
FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(catch2)

add_executable(autograd_tests
        test_placeholder.cpp
        value_test.cpp
)
target_link_libraries(autograd_tests PRIVATE autograd Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(autograd_tests)