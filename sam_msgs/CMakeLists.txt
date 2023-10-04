cmake_minimum_required(VERSION 3.8)
project(sam_msgs)

set(sam_msgs_MAJOR_VERSION 0)
set(sam_msgs_MINOR_VERSION 1)
set(sam_msgs_PATCH_VERSION 0)
set(sam_msgs_VERSION
  ${sam_msgs_MAJOR_VERSION}.${sam_msgs_MINOR_VERSION}.${sam_msgs_PATCH_VERSION})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArrayCommand.msg"
  "msg/BallastAngles.msg"
  "msg/CircuitStatus.msg"
  "msg/CircuitStatusStampedArray.msg"
  "msg/CircuitStatusStamped.msg"
  "msg/Command.msg"
  "msg/ConsumedChargeArray.msg"
  "msg/ConsumedChargeFeedback.msg"
  "msg/ConsumedCharge.msg"
  "msg/CTP.msg"
  "msg/Leak.msg"
  "msg/LightCommandArray.msg"
  "msg/LightCommand.msg"
  "msg/PercentStamped.msg"
  "msg/ThrusterAngles.msg"
  "msg/ThrusterRPMs.msg"
  "msg/UavcanNodeStatus.msg"
  "msg/UavcanNodeStatusNamedArray.msg"
  "msg/UavcanNodeStatusNamed.msg"
  "srv/UavcanUpdateBattery.srv"
  DEPENDENCIES geometry_msgs sensor_msgs std_msgs
 )