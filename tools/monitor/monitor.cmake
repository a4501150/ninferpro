# infernix-monitor, the status-page sidecar (tools/monitor/README.md). Included from
# apps/CMakeLists.txt; Linux only (NVML through dlopen, /proc).

if(WIN32)
  return()
endif()

set(INFERNIX_MONITOR_DIR "${PROJECT_SOURCE_DIR}/tools/monitor")
set(INFERNIX_MONITOR_WEB
  "${INFERNIX_MONITOR_DIR}/web/index.html"
  "${INFERNIX_MONITOR_DIR}/web/app.js"
  "${INFERNIX_MONITOR_DIR}/web/app.css")
set(INFERNIX_MONITOR_WEB_HEADER "${CMAKE_BINARY_DIR}/generated/monitor_web.h")
add_custom_command(
  OUTPUT "${INFERNIX_MONITOR_WEB_HEADER}"
  COMMAND ${CMAKE_COMMAND} -DOUT=${INFERNIX_MONITOR_WEB_HEADER} -DWEB_DIR=${INFERNIX_MONITOR_DIR}/web
          -P ${INFERNIX_MONITOR_DIR}/embed_web.cmake
  DEPENDS ${INFERNIX_MONITOR_WEB} "${INFERNIX_MONITOR_DIR}/embed_web.cmake"
  COMMENT "Embedding the infernix-monitor page"
  VERBATIM)

# The readers and the state, apart from main.cpp so the tests link them.
add_library(infernix_monitor_core STATIC
  "${INFERNIX_MONITOR_DIR}/prometheus.cpp"
  "${INFERNIX_MONITOR_DIR}/file_tail.cpp"
  "${INFERNIX_MONITOR_DIR}/records.cpp"
  "${INFERNIX_MONITOR_DIR}/system_sampler.cpp"
  "${INFERNIX_MONITOR_DIR}/control.cpp"
  "${INFERNIX_MONITOR_DIR}/monitor.cpp")
target_compile_features(infernix_monitor_core PUBLIC cxx_std_20)
# nvml.h only: NVML itself is loaded at run time.
target_include_directories(infernix_monitor_core
  PUBLIC "${INFERNIX_MONITOR_DIR}"
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(infernix_monitor_core
  PUBLIC infernix::json infernix::httplib Threads::Threads ${CMAKE_DL_LIBS})

add_executable(infernix-monitor "${INFERNIX_MONITOR_DIR}/main.cpp" "${INFERNIX_MONITOR_WEB_HEADER}")
target_include_directories(infernix-monitor PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(infernix-monitor PRIVATE infernix_monitor_core)
