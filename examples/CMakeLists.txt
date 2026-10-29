# ---------------------------------------------------------------------------
# examples —— 开发期的试验场（源码内，直接链接内部 target mcn_core）
#
#   * 根 CMakeLists 里 add_subdirectory(examples)，与库一起增量构建；
#     改头文件后立刻生效，没有“install 过期 -> 链接到旧库”的陷阱。
#   * 本目录下每个 *.cpp / *.cu 自动成为一个同名可执行文件
#     （CONFIGURE_DEPENDS：新增文件会自动重新 configure）。
#   * 需要写 kernel（<<<>>>）的文件必须以 .cu 结尾，.cpp 里 nvcc 不接受。
#
#   构建：cmake --build --preset debug
#   产物：build/bin/<Debug|Release>/<文件名>.exe
#
#   注意：这里用 target 名 mcn_core（内部名），不是 mcn::mcn_core（导出名）。
#   后者是给仓库之外、通过 find_package(mcn) 使用本库的项目准备的。
# ---------------------------------------------------------------------------
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

if(NOT EXAMPLE_SOURCES)
    message(STATUS "examples/: 暂无源文件")
endif()

foreach(src IN LISTS EXAMPLE_SOURCES)
    get_filename_component(name "${src}" NAME_WE)

    if(TARGET ${name})
        message(WARNING "examples/: 跳过 ${src}，target '${name}' 已存在")
        continue()
    endif()

    add_executable(${name} "${src}")
    # mcn_core 的 include 目录与 CUDA::cudart 都是 PUBLIC，自动传递下来。
    target_link_libraries(${name} PRIVATE mcn_core)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()
