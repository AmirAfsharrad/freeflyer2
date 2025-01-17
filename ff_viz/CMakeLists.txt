cmake_minimum_required(VERSION 3.8)
project(ff_viz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find system dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(ff_msgs REQUIRED)

add_executable(renderer_node src/renderer_node.cpp)
target_include_directories(renderer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(renderer_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(renderer_node
  geometry_msgs
  rclcpp
  tf2_ros
  visualization_msgs

  ff_msgs
)

install(TARGETS renderer_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # disable Python style checkers
  set(AMENT_LINT_AUTO_EXCLUDE ament_cmake_flake8 ament_cmake_pep257)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY model launch rviz
        DESTINATION share/${PROJECT_NAME})

ament_package()
