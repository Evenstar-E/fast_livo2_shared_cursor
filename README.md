# FAST-LIVO2 Recovery Research

基于 FAST-LIVO2 的退化检测与历史地图恢复研究代码精简版。当前主线是 **LiDAR 测距尺度故障**。

> 研究原型，`paper_ready=false`。已有部分地图和平均位置改善，但还没有通过地图、轨迹、覆盖与当前 clean 安全的联合验收。不是已经验证的通用自动恢复系统。

## 包含内容

|目录|内容|
|---|---|
|`src/`、`include/`|主工程实际编译依赖：LIO/VIO、故障注入、状态与地图 checkpoint、原始输入回放、候选验证、输出 mux|
|`scripts/learned_selector/`|时序模型、特征与动作模型研究模块，包括 GRU、因果注意力和相关变体；保留文件名便于追溯|
|`scripts/`|独立评测、私有候选评审和主工程需要的工具|
|`config/`、`launch/`、`rviz_cfg/`|传感器配置及普通 ROS 启动入口|
|`tests/`|18 个 C++ 核心回归、2 个 Python 评测/候选回归|
|`docs/`|架构、复现边界、研究结果（含负面结果）|

不包含 bag、地图、载荷、模型权重、编译输出、个人路径收据、上千份一次性调试脚本或 Git 历史。原开发工作区未删除或改写。

## 状态与重要区别

本包主工程来自当前 `src/include` 工作树；**并非最新 df1bed 或 94731 隔离实验二进制的独立复现包**。那些实验通过历史冻结源码和生成文件构建，不能用本目录普通 catkin 构建冒充。最新有界交接策略已失败，未作为新默认算法加入本包。学习模块也未在此精简包中与 ROS 启动链路重新集成；没有权重不能声称可重现冻结选择器结果。

详见 [架构](docs/ARCHITECTURE.md)、[复现与数据](docs/REPRODUCIBILITY.md)、[研究状态](docs/RESEARCH_STATUS.md)。

## 构建主工程

目标环境：Ubuntu 20.04 / ROS Noetic，C++17；依赖 PCL、Eigen、OpenCV、旧版非模板 Sophus、rpg_vikit、OpenSSL、Boost。上游依赖说明保留于 [UPSTREAM_README.md](UPSTREAM_README.md)。

将本目录放到 `~/catkin_ws/src/fast_livo`，安装依赖后：

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin_make -DCATKIN_ENABLE_TESTING=ON
source devel/setup.bash
cd build/fast_livo
ctest --output-on-failure
```

本次打包验证结果见 `docs/VALIDATION.md`；没有完成的完整 ROS 构建不能视为已通过。

普通 HILTI 启动（使用你自行取得的 bag 与匹配标定）：

```bash
roslaunch fast_livo mapping_hesaixt32_hilti22.launch rviz:=false
# 另一终端加载同一 ROS 环境后：rosbag play /absolute/path/to/data.bag
```

这条入口是普通估计器，不是私有六臂实验队列；HILTI 配置默认包含视觉，纯 LIO 需在独立配置中关闭图像输入。

## 轻量回归

```bash
python -m pip install -r requirements-eval.txt
python tests/evaluation_metrics_test.py
python tests/shadow_candidate_supervisor_test.py
```

学习模块额外需要 `requirements-models.txt`；该文件仅声明依赖，不是已冻结的训练环境。

## 来源与许可

基于 [hku-mars/FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)，保留原作者声明、代码头及 LICENSE。上游 README 声明 GPLv2，继承的 package.xml 仍写 BSD；本次不擅自重新许可，差异在 [NOTICE.md](NOTICE.md) 明确记录。

代码导出逐文件来源与 SHA256 见 `SOURCE_MANIFEST.json`。发布前阅读 [上传说明](docs/PUBLISHING.md)。
