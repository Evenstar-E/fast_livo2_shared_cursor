# 架构与范围

1. `preprocess` / `IMU_Processing`：读取 LiDAR/IMU，预处理、传播与去畸变。
2. `voxel_map` / `LIVMapper`：几何匹配、状态更新和地图维护；VIO 模块仍保留，避免破坏原主工程依赖。
3. `fault_injection_model` / `pointcloud_fault_injector`：对输入副本注入可控退化，原 bag 保持不变。
4. checkpoint、mutation journal、replay payload：为隔离候选恢复状态并重处理历史输入。
5. 几何、健康尾、收益与候选仲裁：检查候选是否值得采用；拒绝与实际修复是不同结局。
6. `shadow_output_mux`：公开输出的来源管理；存在代码不等于当前版本已验收。
7. `scripts/learned_selector`：学习选择器的独立研究模块，不是上述主工程已集成的最新冻结推理入口。

代码按生产编译目标及其本地 include 依赖收集；没有重命名算法、合并历代状态机或修改估计器公式。只缩减发布版 CMake 的测试目标，生产目标保持原文。
