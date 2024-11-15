project("Tests")

include(FetchContent)

# Clone googleTest 
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)

if(WIN32)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(${PROJECT_NAME} tests.cpp)

target_link_libraries(${PROJECT_NAME} GTest::gtest)

# set_target_properties(${PROJECT_NAME} PROPERTIES 
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}/"
# )

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
