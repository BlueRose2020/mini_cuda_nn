# DQN 贪吃蛇：多文件示例
#
#   snake.*  环境（网格、碰撞、11 维特征）
#   agent.*  DQN（回放池、目标网络、epsilon-greedy、一次 SGD 更新）
#   main.cpp 训练 / 评估 / 演示循环
add_executable(dqn_snake
    main.cpp
    snake.cpp
    agent.cpp
)

target_link_libraries(dqn_snake PRIVATE mcn_core)

set_target_properties(dqn_snake PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
