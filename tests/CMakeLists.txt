add_executable(test_print test_print.cpp)
target_link_libraries(test_print PRIVATE mcn_core)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE mcn_core)

add_test(NAME test_tensor COMMAND test_tensor)