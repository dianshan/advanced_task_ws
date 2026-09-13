# Advanced Task Workspace

`advanced_task_ws` 是螺母穿轴装配任务的当前 ROS 2 工作空间。当前包含
`axis_perception` 包，用于从 Orbbec RGB-D 数据中用几何规则识别固定无螺纹柱，
并将柱顶中心输出到机器人基座坐标系。

## 目录结构

```text
advanced_task_ws/
├── README.md
├── src/
│   └── axis_perception/
       ├── axis_perception/
       │   ├── axis_detector.py       # 纯几何检测算法，可离线测试
       │   └── axis_detector_node.py  # ROS 2 节点
       ├── config/
       │   └── axis_detector.yaml     # 话题、坐标系和检测参数
       ├── launch/
       │   └── axis_detector.launch.py
       └── test/
           └── test_axis_detector.py
├── build/                            # colcon 构建产物
├── install/                          # ROS 2 运行环境
└── log/                              # 构建日志
```

`Dexterous-Hand`、`orbbec_ws` 和 `linkerbot_ws` 位于本工作空间的同级目录，分别
提供机器人依赖、相机依赖和基础任务参考实现。

## 识别流程

`axis_perception` 不使用 YOLO，而是利用无螺纹柱的固定几何特征：

1. 订阅对齐后的彩色图、深度图和相机内参。
2. 将有效深度像素反投影为相机坐标系点云。
3. 对工作区点云做 RANSAC 平面拟合，估计桌面平面。
4. 以桌面为高度基准，筛选高度在 `0.105 m ~ 0.185 m` 的柱顶点。
5. 用柱顶点估计柱心、半径和圆度。
6. 按柱高、半径、圆度和内点数量校验候选。
7. 通过 TF 将柱顶中心从相机光学坐标系转换到 `base_link`。
8. 连续 5 帧中心抖动和高度抖动满足阈值后锁定结果。

无螺纹柱在整轮任务中固定，因此一旦锁定，节点会持续输出该结果，不需要每帧重新识别。

## 输入话题

默认话题如下，可在 `config/axis_detector.yaml` 中修改：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `color_topic` | `/camera/color/image_raw` | 彩色图像 |
| `depth_topic` | `/camera/aligned_depth_to_color/image_raw` | 与彩色图对齐的深度图 |
| `depth_info_topic` | `/camera/aligned_depth_to_color/camera_info` | 对齐深度流内参 |

需要相机驱动输出 RGB-D 对齐流。Orbbec ROS 2 驱动通常需要开启
`depth_registration: true`。

## 输出话题

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/axis_detection/pose` | `geometry_msgs/msg/PoseStamped` | 柱顶中心，默认 `base_link` 坐标系 |
| `/axis_detection/debug` | `sensor_msgs/msg/Image` | 标注柱心和半径的调试图像 |
| `/axis_detection/status` | `std_msgs/msg/String` | JSON 状态，包含 `searching`、`candidate`、`locked` |

查看状态示例：

```bash
ros2 topic echo /axis_detection/status
```

查看调试图像：

```bash
ros2 run rqt_image_view rqt_image_view /axis_detection/debug
```

## 构建与运行

`advanced_assembly` 依赖 `Dexterous-Hand` 提供的 `lbot_arm_interfaces`。构建进阶工作空间前，先让当前 shell 找到该接口包；以下命令假设 `advanced_task_ws`、`Dexterous-Hand`、`orbbec_ws` 和 `linkerbot_ws` 位于同一父目录：

```bash
source /opt/ros/jazzy/setup.bash
source ../Dexterous-Hand/install/setup.bash
colcon build --packages-select advanced_assembly axis_perception --symlink-install
source install/setup.bash
```

如果 `Dexterous-Hand/install` 未构建，请在 `Dexterous-Hand` 中先执行 `colcon build --packages-select lbot_arm_interfaces`。

在仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
cd advanced_task_ws
colcon build --packages-select axis_perception --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

启动检测节点：

```bash
ros2 launch axis_perception axis_detector.launch.py
```

如果只想验证相机系识别，不执行 TF 转换，可以关闭 TF：

```bash
ros2 launch axis_perception axis_detector.launch.py use_tf:=false
```

如果只想验证代码而不接相机，可运行单元测试：

```bash
cd advanced_task_ws/src/axis_perception
PYTHONPATH=. python3 -m pytest test/test_axis_detector.py -q
```

## TF 要求

默认目标坐标系是 `base_link`。若相机安装在机器人外部，需要提供：

```text
base_link -> camera_color_optical_frame
```

如果 TF 缺失，节点会返回 None，不会把相机坐标当作机器人坐标输出。

建议复用 `linkerbot_ws` 中已有的外参标定结果。运行前可用以下命令检查：

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

## 关键参数

所有参数都在 `src/axis_perception/config/axis_detector.yaml` 中配置。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `workspace_min_x_m` / `workspace_max_x_m` | `-0.10` / `0.10` | 相机系 X 搜索范围 |
| `workspace_min_y_m` / `workspace_max_y_m` | `-0.10` / `0.10` | 相机系 Y 搜索范围 |
| `min_axis_height_m` / `max_axis_height_m` | `0.105` / `0.185` | 柱顶高度接受范围 |
| `cylinder_radius_m` | `0.015` | 柱半径 |
| `cylinder_radius_tolerance_m` | `0.008` | 半径容差 |
| `min_top_points` | `60` | 柱顶最小点数 |
| `min_top_inlier_points` | `80` | 柱顶圆环最小内点数 |
| `min_circularity` | `0.65` | 柱顶点圆度阈值 |
| `lock_after_confirmations` | `5` | 锁定前连续观测次数 |
| `max_center_jitter_m` | `0.008` | 锁定前中心最大标准差 |
| `max_top_height_jitter_m` | `0.015` | 锁定前柱顶高度最大标准差 |

现场相机视角不同时，优先标定 `workspace_*` 四个参数。它们是相机光学坐标系下的
水平搜索范围，不是机器人基座坐标范围。

## 调试建议

1. 先启动相机并确认 RGB 和 aligned depth 话题频率：

   ```bash
   ros2 topic hz /camera/color/image_raw
   ros2 topic hz /camera/aligned_depth_to_color/image_raw
   ```

2. 用 `use_tf:=false` 先确认相机系内可以检测到柱子。

3. 查看调试图像，确认绿色圆是否贴合柱顶轮廓。

4. 再开启 TF，用 rviz2 或 Rerun 对比 `/axis_detection/pose` 与实际点云。

5. 如果状态一直是 `searching`，按顺序检查：
   - 深度图是否为对齐后的话题；
   - `workspace_*` 是否把柱子排除在外；
   - 柱顶有效深度点数是否达到 `min_top_points`；
   - `cylinder_radius_m` 是否与实际柱径一致；
   - TF 是否可用。

## 安全边界

本包只做感知，不发送机械臂运动命令。真实装配任务应通过任务层状态机消费
`/axis_detection/pose`，并保持运动执行开关默认关闭。首次联调时应使用低速、小幅度
动作，并确认急停可用。


## 进阶任务：四螺母穿轴

`advanced_assembly` 将基础任务的螺母识别和左手抓取逻辑迁移到本工作空间，并扩展为四颗目标：

| ID | 标签 | 装配顺序 | 区分方式 |
| --- | --- | --- | --- |
| 1 | `m45_silver` | 1 | M45 尺寸 + 高灰度 |
| 2 | `m45_black` | 2 | M45 尺寸 + 低灰度 |
| 3 | `m33` | 3 | 尺寸区间 |
| 4 | `m27` | 4 | 尺寸区间 |

识别节点迁移了基础任务的黑框检测、内边框 ROI、自适应阈值、black-hat、轮廓几何校验、大块轮廓 watershed 分割和透视尺寸校正；同时取消基础任务的三颗上限，改为四颗候选，并用轮廓灰度区分两个 M45。RGB-D 定位和 TF 转换与基础任务相同，输出到 `base_link`。

### 感知调试

```bash
# 终端 1：外部相机驱动与 base_link -> camera TF
# 终端 2：只启动四螺母与立柱识别
ros2 launch advanced_assembly advanced_task.launch.py
ros2 topic echo /advanced_nut_sequence
ros2 topic echo /axis_detection/status
```

调试图像话题：

- `/advanced_nut_detection/debug`
- `/axis_detection/debug`

四颗螺母需要在 3 秒窗口内连续稳定 3 帧。全流程模式由控制器在每颗开始和释放完成后调用序列事件服务；每完成一颗后，节点期望剩余 3、2、1 颗重新稳定，不会因为短暂漏检而自动把下一颗标记完成。手工调试时也可以直接调用：

```bash
ros2 service call /advanced_nut_sequence/event/start std_srvs/srv/Trigger {}
ros2 service call /advanced_nut_sequence/event/complete std_srvs/srv/Trigger {}
ros2 service call /advanced_nut_sequence/event/retry std_srvs/srv/Trigger {}
ros2 service call /advanced_nut_sequence/event/reset std_srvs/srv/Trigger {}
```

### 单颗螺母调试

先填写 `src/advanced_assembly/config/advanced_task.yaml` 中的现场参数：

- `middle_position`：人工测量的中转 TCP 位置 `base_link [x,y,z]`；
- `left_enter_route` / `left_home_route`：左臂示教关节路线；
- `right_enter_route` / `right_home_route`：右臂示教关节路线；
- `left_grasp_rpy` / `right_grasp_rpy`：实测 Arm_Tip RPY；
- `left_tcp_offset` / `right_tcp_offset`：Arm_Tip 到抓取 TCP 的工具系偏移；
- `hand_*`：O6 手参数；
- 关节限位和速度。

目标螺母、中转位和轴上位置都按抓取 TCP 计算，控制器再使用与基础任务相同的旋转公式把它们转换为驱动接口需要的 Arm_Tip 位置。`left_tcp_offset` 可从 `linkerbot_ws/config/control/nut_task.yaml` 的实测 O6 值开始，右手偏移必须另行测量。

立柱上方参数全部独立可调：

- `axis_above_height`：相对识别柱顶的上方接近高度，默认 80 mm；
- `axis_descend_height`：穿轴下降终点，相对识别柱顶默认 20 mm；
- `axis_release_lift_height`：松手后撤离高度，默认 100 mm。

无运动配置检查：

```bash
ros2 launch advanced_assembly advanced_task.launch.py \
  run_task:=true execute_task:=false task_mode:=validate
```

执行指定单颗（示例为 M45 银色，`target_id` 依次为 1、2、3、4）：

```bash
ros2 launch advanced_assembly advanced_task.launch.py \
  run_task:=true execute_task:=true task_mode:=single target_id:=1
```

单颗流程为：等待立柱和当前目标 → 左臂进入 → 左手按基础任务方式抓取 → 提升并放到中转位 → 左臂回位 → 右臂进入 → 右手从中转位抓取 → 移到轴上方 → 下降 → 松手 → 提升 → 右臂回位。

### 全流程调试

```bash
ros2 launch advanced_assembly advanced_task.launch.py \
  run_task:=true execute_task:=true task_mode:=full
```

全流程复用同一个 `run_single()` 代码路径，按 ID 1→2→3→4 依次执行。每颗开始时发送 `start`，右手释放并回位后发送 `complete`，检测节点随后按剩余 3、2、1 颗重新稳定分类。真实运动前必须把配置中的六个校准开关全部改为 `true`；这些开关和 `execute_task` 默认均为 `false`。

任务状态：

```bash
ros2 topic echo /advanced_assembly/status
```

### 安全限制

右手轴上方的三个高度参数只做终点几何计算，当前任务层不做碰撞规划、连续路径 IK 校验和抓取成功率视觉确认。首次实机调试应使用低速，确认左右臂工作空间无交叉风险，并保持急停可用。运动异常时控制器会向双手发送张开命令并分别请求左/右臂急停；无法通信或强制终止时无法保证命令送达。
