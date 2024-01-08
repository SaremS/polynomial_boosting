cmake_minimum_required(VERSION 3.4...3.18)

project(polynomial_boosting
	VERSION 0.1
	DESCRIPTION "Boosting with polynomial regression CART stumps"
	LANGUAGES CXX
)


include_directories(include)
include(FetchContent)

FetchContent_Declare(
  Eigen	
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG        3.4
)

FetchContent_MakeAvailable(Eigen)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.11
)

FetchContent_MakeAvailable(pybind11)


pybind11_add_module(
	polynomial_boosting
	lib/tree/treestump.cpp
	lib/linalg.cpp
	lib/loss_functions/quadratic_loss.cpp
	lib/regression_models/linear_regression_model.cpp
	lib/gradient_boosting/gradient_boosting.cpp
	lib/gradient_boosting/goss_sampler.cpp
	lib/seedable_rng.cpp
)

target_link_libraries(polynomial_boosting PRIVATE Eigen3::Eigen)

target_compile_definitions(polynomial_boosting
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

enable_testing()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_executable(
	unittest_quadratic_loss
	include/loss_functions/loss_function.h
	include/loss_functions/quadratic_loss.h
	lib/loss_functions/quadratic_loss.cpp
	include/linalg.h
	lib/linalg.cpp
	tests/unittest_quadratic_loss.cpp
)

target_link_libraries(unittest_quadratic_loss gtest_main Eigen3::Eigen)

add_executable(
	unittest_linalg
	include/linalg.h
	lib/linalg.cpp
	tests/unittest_linalg.cpp
)

target_link_libraries(unittest_linalg gtest_main Eigen3::Eigen pybind11::embed)

add_executable(
	unittest_linear_regression_model
	include/linalg.h
	lib/linalg.cpp
	include/regression_models/regression_model.h
	include/regression_models/linear_regression_model.h
	lib/regression_models/linear_regression_model.cpp
	include/regression_models/fast_linear_regression.h
	lib/regression_models/fast_linear_regression.cpp
	tests/unittest_linear_regression_model.cpp
)

target_link_libraries(unittest_linear_regression_model gtest_main Eigen3::Eigen pybind11::embed)

add_executable(
	unittest_treestump
	include/linalg.h
	lib/linalg.cpp
	include/regression_models/regression_model.h
	include/regression_models/linear_regression_model.h
	lib/regression_models/linear_regression_model.cpp
	include/tree/treestump.h
	lib/tree/treestump.cpp
	include/loss_functions/loss_function.h
	include/loss_functions/quadratic_loss.h
	lib/loss_functions/quadratic_loss.cpp
	include/seedable_rng.h
	lib/seedable_rng.cpp
	include/gradient_boosting/goss_sampler.h
	lib/gradient_boosting/goss_sampler.cpp
	tests/unittest_treestump.cpp
)

target_link_libraries(unittest_treestump gtest_main Eigen3::Eigen pybind11::embed)

add_executable(
	unittest_gradient_boosting
	include/linalg.h
	lib/linalg.cpp
	include/regression_models/regression_model.h
	include/regression_models/linear_regression_model.h
	lib/regression_models/linear_regression_model.cpp
	include/tree/treestump.h
	lib/tree/treestump.cpp
	include/loss_functions/loss_function.h
	include/loss_functions/quadratic_loss.h
	lib/loss_functions/quadratic_loss.cpp
	include/gradient_boosting/gradient_boosting.h
	lib/gradient_boosting/gradient_boosting.cpp
	include/seedable_rng.h
	lib/seedable_rng.cpp
	include/gradient_boosting/goss_sampler.h
	lib/gradient_boosting/goss_sampler.cpp
	tests/unittest_treestump.cpp
)

target_link_libraries(unittest_gradient_boosting gtest_main Eigen3::Eigen pybind11::embed)


include(GoogleTest)
gtest_discover_tests(unittest_quadratic_loss)
gtest_discover_tests(unittest_linalg)
gtest_discover_tests(unittest_linear_regression_model)
gtest_discover_tests(unittest_treestump)
gtest_discover_tests(unittest_gradient_boosting)
