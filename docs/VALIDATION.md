# 导出验证（2026-10-28）

本次没有训练、bag 播放或算法修改。

|检查|结果|边界|
|---|---|---|
|140份复制文件 SHA256 对原工作树|PASS|源码原样；CMake 测试配置和发布文档为新生成|
|CMake 显式源文件、四份 launch 内本包文件引用|PASS|不是完整链接或 ROS 启动验证|
|所有导出的 Python 语法|PASS|学习模块没有加载权重或完成推理验证|
|常见私钥／token／密码赋值模式扫描|未发现|启发式扫描，不是全面安全证明；保留上游作者联系方式与原注释|
|evaluation_metrics_test.py|PASS|评测公式与人工回归数据，不是实际恢复性能|
|shadow_candidate_supervisor_test.py|PASS|候选判断回归，不是公开 mux 验收|
|10项轻量 C++ 编译与运行|10/10 PASS|WSL g++、C++17、O0、断言启用|
|完整 catkin/ROS 编译及其余8项 C++ 测试|NOT_RUN|需要完整依赖环境，未以轻量测试替代|
|当前学习选择器端到端接线、训练复现|NOT_RUN|本精简包不含冻结权重、训练数据与历史生成构建|

C++ 已运行：fault_injection_model、range_scale_correction_estimator、map_transaction_state、online_harm_detector、healthy_tail_validator、recovery_proposal_arbiter、recovery_benefit_validator、selective_residual_gate、robust_residual_weight、shadow_generation_selector 对应测试。

Windows Python 初次测试遇到沙箱临时目录写权限问题；指定工作区内临时目录后，原测试未改即通过。C++首次调度没有找到验证脚本，确认脚本落盘后执行通过；不是物理实验重试。

`SHA256SUMS` 覆盖打包时的文件（自身除外），用于传输后核验；不包含原始实验数据，因此不能替代科学证据核验。
