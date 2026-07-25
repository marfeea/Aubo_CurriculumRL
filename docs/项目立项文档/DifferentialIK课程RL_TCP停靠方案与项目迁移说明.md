# DifferentialIK + 课程 RL TCP 停靠方案与项目迁移说明

> 文档用途：为新工作区中的 TCP 停靠方案提供当前项目基线、跨项目迁移清单和粗粒度 PRD。
>
> 当前事实来源：`configs/`、`tasks/WithClaw/`、`tasks/common/`、`tools/` 及现有项目文档。
>
> 边界：本文描述迁移和方案设计，不代表新工作区已经完成实现或仿真验收。

## 1. 当前 TCP 项目基线

### 1.1 项目结构

当前 TCP 停靠任务位于 `tasks/WithClaw/`，与旧版无夹爪任务 `tasks/WithoutClaw/` 隔离：

```text
Test/
├─ configs/                    # 全项目资产、场景、碰撞、相机及控制器配置
├─ tasks/
│  ├─ common/                  # 任务无关的 SB3 运行、输出路径和训练统计
│  ├─ WithoutClaw/             # 旧版无夹爪 Flange reach 基线
│  └─ WithClaw/                # 当前带夹爪 TCP 停靠任务
│     ├─ task_cfg.py           # TCP 标定、目标状态、阈值和奖励权重
│     ├─ asset_cfg.py          # 带夹爪 AUBO articulation 配置
│     ├─ scene_cfg.py          # WithClaw 场景组装
│     ├─ env_cfg.py            # ManagerBasedRLEnvCfg 总装
│     ├─ observations.py       # 观测与现有 Lula IK 动作配置
│     ├─ tcp.py                # TCP 世界/根坐标运动学
│     ├─ orientation.py        # 工具轴、目标轴和期望姿态计算
│     ├─ reset_state.py        # 离散目标状态选择与批量 reset 数据
│     ├─ events.py             # Reset 事件和逐环境缓存初始化
│     ├─ mdp_logic.py          # 可独立测试的停靠状态机与奖励纯逻辑
│     ├─ runtime.py            # Isaac 场景数据到任务状态的适配
│     ├─ rewards.py            # 奖励项组装
│     ├─ terminations.py       # 成功、失败和超时终止
│     ├─ collision_cfg.py      # WithClaw 机器人接触传感器
│     ├─ train.py              # SB3 PPO 训练入口
│     └─ eval.py               # Checkpoint 评估入口
├─ tools/
│  ├─ ik.py                    # 当前三维动作到 Lula IK 的适配
│  ├─ lula_ik.py               # 当前 Lula IK 控制器封装
│  ├─ scene.py                 # 场景查询、位姿写入和坐标辅助
│  └─ contact.py               # 接触传感器兼容读取
├─ source/Test/                # 纯逻辑、坐标、奖励和回调测试
├─ checkpoints/                # 当前训练模型产物
└─ docs/                       # 设计、施工和验收记录
```

### 1.2 当前管理方法

- 任务按 `WithClaw`、`WithoutClaw` 隔离，只有任务无关能力进入 `tasks/common`。
- 环境采用 Isaac Lab Manager-Based 结构，场景、动作、观测、奖励、事件和终止条件分模块组装。
- TCP 标定、目标状态、停靠阈值和奖励权重集中在 `tasks/WithClaw/task_cfg.py`。
- Reset 事件原子更新目标位姿、preposition、期望法兰姿态及逐环境历史缓存。
- 停靠判定采用带进入/退出迟滞的状态机，且同一控制步只允许提交一次状态更新。
- 当前策略输出三维笛卡尔位置增量；`tools/ik.py` 使用 Lula 非线性 IK 生成关节位置目标，并在接近目标后渐进启用姿态约束。
- 训练采用 SB3 PPO；日志和 Checkpoint 按 `<任务名>/sb3_aubo` 分目录管理。
- 现有改造过程采用“阶段施工、验证、用户验收、再进入下一阶段”的状态门禁。

### 1.3 当前任务真值

以下条件属于迁移时应保持稳定的任务语义，不应与具体 IK 实现绑定：

- 控制机器人：`AUBObot`；场景同时加载 `AUBObot_2`，但策略只控制第一台机器人。
- 末端刚体：`Flange`。
- 停靠目标资产：`ws_interactive_reagent_01_sample_bottle`。
- TCP 进入距离：`0.04 m`。
- TCP 退出距离：`0.055 m`，用于构成迟滞区间。
- TCP 速度阈值：`0.03 m/s`。
- 工具轴姿态误差阈值：`10°`。
- 连续停留要求：`2` 个控制步。
- 仿真频率：`120 Hz`；控制 decimation 为 `30`，策略频率为 `4 Hz`。
- Episode 时长：`40 s`，即最多 `160` 个策略步。
- 失败条件：TCP 越出工作空间、非法接触力超过阈值、目标位移过大、目标速度过高以及超时。

当前代码配置了 4 个离散目标初始状态；旧文档中的“五状态”描述不是当前 `TARGET_INITIAL_STATES` 的实际数量。

## 2. 新工作区迁移原则

1. 迁移任务事实和可验证逻辑，不整体复制当前项目的历史结构。
2. 资产路径、位姿、阈值和课程参数必须具有单一事实来源。
3. 依赖方向保持为“配置 → 纯逻辑 → Isaac 适配 → 训练与数据输出”。
4. 先建立与当前项目等价的场景和成功判据，再替换控制器并引入课程学习。
5. 最终评估条件保持固定；课程阶段只能改变训练分布、辅助奖励或约束启用顺序。
6. 当前 Lula 方案保留为对照基线，不迁移为新方案的运行依赖。

## 3. 必须迁移的跨项目内容

### 3.1 外部资产清单与路径管理

当前资产根目录硬编码为：

```text
D:/project/S2R/Asset
```

新工作区不应继续在多个文件中写绝对路径。建议只保留一个 `ASSET_ROOT` 配置入口，通过环境变量或工作区本地配置解析，并在启动时校验必需文件。

必须可访问的资产包括：

| 资产 | 当前相对路径 | 用途 | 迁移要求 |
|---|---|---|---|
| 带夹爪 AUBO | `AUBO_E5/AUBO_E5_Withclaw.usd` | 受控机器人 | 必须迁移或建立稳定外部引用 |
| 实验室环境 | `Laboratory/M_Laboratory.usd` | 背景环境 | 可选；不影响核心控制时可延后加载 |
| 工作站拆分资产根目录 | `QKL-HX-300-II-00/Part/` | 工作站、托盘和交互物体 | 必须迁移停靠相关子集 |
| 样品瓶目标 | `QKL-HX-300-II-00/Part/Reagent_01/M_Reagent_01.usd` | TCP 停靠目标 | 必须迁移 |

注意事项：

- 当前训练场景使用 `configs/place_cfg.py` 按零件加载拆分 USD，而不是依赖 `WorkStation_All.usd` 整体加载。
- USD 内部引用的纹理、材质、子 USD 和碰撞定义必须随资产一起迁移，不能只复制顶层 USD。
- 仓库 `.gitignore` 明确排除 USD；新工作区应使用外部资产目录、软链接、资产包或部署脚本管理资产。
- `tasks/WithClaw/task_cfg.py` 当前重复定义了机器人绝对路径。迁移时应消除该重复，以新工作区资产清单为唯一来源。

### 3.2 USD 和 articulation 接口契约

新工作区加载机器人资产后，必须验证以下名称和结构保持有效：

```text
场景实体名：AUBObot
第二机器人实体名：AUBObot_2
articulation prim：AUBO_E5
末端刚体：Flange
机械臂关节：Joint1, Joint2, Joint3, Joint4, Joint5, Flange
夹爪关节：UpperFinger, DownFinger
```

Differential IK 依赖 Jacobian 中的 body id 和关节顺序。不能只凭字符串列表假设顺序正确；场景初始化后应通过 `SceneEntityCfg.resolve()` 或等价机制解析实际关节和 body id，并检查：

- 六个机械臂关节均被唯一匹配；
- `Flange` 是可读取位姿、线速度、角速度和 Jacobian 的刚体；
- 固定基座与浮动基座 Jacobian 索引差异已正确处理；
- 夹爪关节不进入机械臂 Differential IK 求解向量；
- 机器人 USD 启用了接触传感器。

### 3.3 场景放置与坐标约定

当前工作站基准位姿为：

```text
位置：(1.3, 0.0, 0.0) m
旋转：(0.70711, 0.0, 0.0, -0.70711)，四元数顺序为 wxyz
```

机器人基座首先使用工作站局部坐标配置，再由 `workstation_local_to_world_pos()` 派生世界坐标：

| 机器人 | 工作站局部位置 |
|---|---|
| `AUBObot` | `(0.034, -0.013, 0.816)` |
| `AUBObot_2` | `(-0.81, 0.21, 0.816)` |

迁移时必须保留以下坐标约定：

- `W`：Isaac 世界坐标系。
- `E`：每个并行环境的局部坐标系，世界坐标由 `env_origins` 平移得到。
- `B`：第一台机器人的根坐标系。
- `F`：`Flange` 刚体局部坐标系。
- `T`：任务定义的 TCP/工具坐标系。
- 所有四元数均使用 `wxyz`，不得与 `xyzw` 混用。
- 目标状态中的 `pos` 和 `preposition` 是环境局部坐标；写入仿真前需要加对应 `env_origin`。
- 新 Differential IK 动作建议统一表达在 `B` 坐标系，避免并行环境世界偏移进入策略动作。

### 3.4 机器人初始状态和执行器配置

当前机械臂初始关节角为：

| 关节 | 初始值 |
|---|---:|
| `Joint1` | `0°` |
| `Joint2` | `-30°` |
| `Joint3` | `70°` |
| `Joint4` | `45°` |
| `Joint5` | `90°` |
| `Flange` | `0°` |
| `UpperFinger` | `0.0115 m` |
| `DownFinger` | `0.0 m` |

当前执行器基线：

- 机械臂：隐式位置执行器，`effort_limit_sim=2400`、`velocity_limit_sim=3.14`、`stiffness=6000`、`damping=600`。
- 夹爪：`effort_limit_sim=50`、`stiffness=1000`、`damping=50`。
- articulation：关闭自碰撞，位置求解迭代 8 次，速度求解迭代 1 次。

这些参数应先原样迁移用于等价性验证。Differential IK 稳定后，再通过独立实验调整执行器增益，不能把控制器问题与执行器参数变化混在同一次迁移中。

### 3.5 TCP 标定和姿态契约

当前 TCP 使用固定法兰局部偏置：

```text
FLANGE_TO_TCP_TRANSLATION_F = (0.0, -0.12, 0.102) m
FLANGE_TO_TOOL_ROTATION_F   = (sqrt(0.5), 0.0, -sqrt(0.5), 0.0)
TARGET_TO_TOOL_ROTATION_T   = (sqrt(0.5), sqrt(0.5), 0.0, 0.0)
工具前向轴                  = (0.0, 0.0, 1.0)
目标停靠轴                  = (0.0, -1.0, 0.0)
```

必须迁移并保留测试的通用计算包括：

- 根据法兰位姿和局部偏置计算 TCP 世界位置；
- 根据法兰线速度、角速度和偏置切向项计算 TCP 世界线速度；
- 将 TCP 位置和完整相对速度转换到机器人根坐标系；
- 根据目标姿态计算期望法兰姿态；
- 计算工具轴与目标停靠轴的夹角和对齐得分。

当前 TCP 平移偏置是项目确认的第一版近似值，仍属于待实测复核的标定风险。新项目应把它保留为单独可替换的标定参数，不得写入 Differential IK 或奖励实现内部。

### 3.6 目标状态与 Reset 数据流

当前目标资产场景 key 为：

```text
ws_interactive_reagent_01_sample_bottle
```

当前 4 个离散状态如下，位置单位为米，旋转为 `wxyz`：

| 状态 | 目标位置 `pos` | 目标旋转 `rot` | 停靠点 `preposition` |
|---|---|---|---|
| `sample_bottle_state_01` | `(1.537, 0.203, 0.94)` | `(0, 0, 0, 1)` | `(1.537, 0.083, 0.94)` |
| `sample_bottle_state_02` | `(0.91167, 0.1753, 0.96789)` | `(0.70710678, 0, 0, -0.70710678)` | `(1.03167, 0.1753, 0.96789)` |
| `sample_bottle_state_03` | `(0.91167, 0.03036, 0.96676)` | `(0.70710678, 0, 0, -0.70710678)` | `(1.03167, 0.03036, 0.96676)` |
| `sample_bottle_state_04` | `(0.91235, -0.18557, 0.99091)` | `(0.70710678, 0, 0, -0.70710678)` | `(1.03235, -0.18557, 0.99091)` |

Reset 必须按选中的 `env_ids` 原子完成：

1. 选择固定或随机目标状态；
2. 将环境局部目标位置转换为世界位置；
3. 写入目标位姿并清零目标速度；
4. 读取仿真实际目标位姿作为本回合基准；
5. 计算并缓存 preposition 和期望法兰姿态；
6. 清理进展、里程碑、姿态、停靠区、dwell 和控制步缓存。

该流程应继续支持部分环境 Reset，不能用一次性全局变量代替逐环境张量。

### 3.7 停靠状态机、安全约束和奖励语义

建议直接迁移 `mdp_logic.py` 中与 Isaac API 解耦的纯逻辑，并在新工作区保留单元测试。最终成功条件为：

```text
进入停车区
AND TCP 速度低于阈值
AND 工具轴姿态满足阈值
AND 连续满足指定控制步数
```

安全约束基线：

| 条件 | 当前阈值 |
|---|---:|
| TCP 根坐标工作空间 | `x,y∈[-0.75,0.75] m`，`z∈[0.20,1.10] m` |
| 非法接触力 | `50 N` |
| 目标最大位移 | `0.03 m` |
| 目标最大线速度 | `0.05 m/s` |
| 忽略接触刚体 | `Base_Link` |

当前奖励由距离进展、历史最佳进展、邻近度、内区停靠质量、低速停车、首次进入里程碑、工具轴进展、最终成功及安全惩罚组成。新方案可根据课程等级调整辅助奖励，但必须满足：

- 最终成功奖励只绑定统一成功状态机；
- 历史最佳进展只奖励新的最短距离，避免往返刷分；
- Reset 必须清空所有历史奖励状态；
- 最终评估不得使用放宽后的课程成功阈值。

### 3.8 碰撞与接触配置

迁移时应区分两个职责：

- 工作站放置配置只描述资产、语义分组和初始位姿，不在代码中重新生成 USD 碰撞几何。
- 碰撞配置负责读取资产已有 CollisionAPI、安装接触传感器和定义任务失败阈值。

当前机器人接触传感器匹配：

```text
{ENV_REGEX_NS}/AUBObot/AUBO_E5/.*
```

迁移后需要验证：

- 机器人 USD spawn 设置 `activate_contact_sensors=True`；
- ContactSensor 正确挂载到新场景；
- 多环境 contact tensor 的环境维和 body 维解析正确；
- `Base_Link` 接触被排除，其余机器人刚体参与非法碰撞判定；
- 工作站拆分 USD 自带的碰撞 schema 在新路径下仍有效；
- 临时禁用碰撞只能针对明确 prim，不能停用整个工作站 prim。

### 3.9 相机、渲染和数据管线

相机、RTX 渲染和 RGB-D 记录不是 Differential IK 课程训练的核心依赖。建议分层迁移：

- 第一阶段不加载三个 CameraSensor，以减少训练显存和初始化时间。
- 场景等价性或可视化验收时迁移 `configs/camera_cfg.py` 和 `configs/RenderCfg.py`。
- 如果后续策略不使用视觉，训练配置应显式将 camera scene entries 设为 `None`。
- 数据目录、截图和 RGB-D 输出属于生成数据，不能反向控制任务逻辑。

### 3.10 训练运行、依赖和产物管理

可复用的通用部分：

- `tasks/common/paths.py`：按任务隔离日志和 Checkpoint。
- `tasks/common/sb3_runtime.py`：SB3 PPO 环境包装、训练和评估主循环。
- `tasks/common/training_callbacks.py`：记录奖励项回报和终止条件比例。
- `scripts/_bootstrap.py` 或任务入口中的项目根路径注入。

迁移时需要改进：

- 新工作区应锁定实际可运行的 Isaac Sim、Isaac Lab、PyTorch、SB3 和 Python 版本。
- 当前扩展 `setup.py` 未显式声明 SB3，不能把现有环境中“已经安装”视为新工作区依赖声明。
- `checkpoints/` 当前未被顶层 `.gitignore` 排除，并已造成大量二进制工作区改动。新工作区必须忽略 `checkpoints/`、`logs/`、`data/` 和临时渲染产物。
- Checkpoint 名称应包含任务、课程等级、随机种子和运行标签，并保存配套配置快照。
- 课程晋级统计应进入统一回调，不应通过解析控制台文本决定。

## 4. 不应迁移为新方案依赖的内容

以下内容可作为行为对照或调试参考，但不应进入 Differential IK 新方案的运行依赖：

- `configs/lula_cfg.py`；
- `configs/lula/aubo_e5.urdf`；
- `configs/lula/aubo_e5_robot_description.yaml`；
- `tools/lula_ik.py`；
- `tools/ik.py::AuboTaskSpaceIKAction`；
- 现有 Lula 姿态渐进锁定控制流程；
- 历史 Checkpoint 和与旧动作空间绑定的策略参数。

原因是新方案应直接使用 Isaac articulation Jacobian 执行 Differential IK，旧 Lula 控制器的 URDF、求解状态和关节解不能构成新策略的隐式依赖。

## 5. 推荐的新工作区结构

```text
NewTcpDocking/
├─ configs/
│  ├─ assets.py                # 唯一资产根目录、USD 清单和 prim 契约
│  ├─ scene.py                 # 工作站、机器人、目标和传感器配置
│  ├─ task.py                  # TCP 标定、目标状态、最终阈值和奖励基线
│  ├─ differential_ik.py       # Differential IK、阻尼和限幅配置
│  ├─ curriculum.py            # 课程等级、晋级/回退规则和训练分布
│  └─ training.py              # PPO 与运行参数
├─ tasks/tcp_docking/
│  ├─ env_cfg.py
│  ├─ observations.py
│  ├─ actions.py               # Isaac Differential IK 适配
│  ├─ events.py
│  ├─ rewards.py
│  ├─ terminations.py
│  └─ curriculum.py            # 课程运行时更新
├─ logic/
│  ├─ tcp_kinematics.py        # 与 Isaac API 解耦的 TCP 数学
│  ├─ orientation.py
│  ├─ parking_state.py
│  └─ curriculum_state.py
├─ runtime/
│  ├─ scene_access.py
│  ├─ contact.py
│  └─ sb3_runtime.py
├─ scripts/
│  ├─ inspect_assets.py
│  ├─ smoke_env.py
│  ├─ train.py
│  └─ eval.py
├─ tests/
│  ├─ test_tcp_kinematics.py
│  ├─ test_orientation.py
│  ├─ test_parking_state.py
│  ├─ test_curriculum.py
│  └─ test_differential_ik_contract.py
└─ docs/
   └─ 本文档
```

## 6. 迁移来源映射

| 当前来源 | 新工作区目标 | 处理方式 |
|---|---|---|
| `configs/asset.py` | `configs/assets.py` | 迁移并改为可移植资产根目录 |
| `configs/place_cfg.py` | `configs/scene.py` 或独立 `workstation.py` | 迁移工作站分组、路径和派生位姿 |
| `configs/scene_cfg.py` | `configs/scene.py` | 迁移场景、机器人初态和执行器基线 |
| `tasks/WithClaw/task_cfg.py` | `configs/task.py` | 迁移 TCP、目标状态和最终判据，消除资产路径重复 |
| `tasks/WithClaw/tcp.py` | `logic/tcp_kinematics.py` | 基本原样迁移并保留纯逻辑测试 |
| `tasks/WithClaw/orientation.py` | `logic/orientation.py` | 基本原样迁移并保留坐标测试 |
| `tasks/WithClaw/mdp_logic.py` | `logic/parking_state.py` | 迁移状态机和任务纯逻辑 |
| `tasks/WithClaw/reset_state.py` | `logic/` 与 `events.py` | 拆分纯状态选择和 Isaac reset 写入 |
| `tasks/WithClaw/runtime.py` | `runtime/` | 适配新场景和 Differential IK 数据流 |
| `tasks/WithClaw/rewards.py` | 新任务 `rewards.py` | 复用最终语义，按课程重新组装 |
| `tasks/WithClaw/terminations.py` | 新任务 `terminations.py` | 保持最终成功/失败定义 |
| `tools/scene.py`、`tools/contact.py` | `runtime/` | 仅迁移新任务实际使用的最小接口 |
| `tasks/common/*` | `runtime/` 和训练入口 | 复用路径、评估和统计能力，增加课程指标 |
| `configs/lula*`、`tools/lula_ik.py`、`tools/ik.py` | 无 | 不迁移为依赖，只保留基线参考 |

## 7. 建议迁移顺序与验收门禁

### 阶段 A：工作区和资产

- 建立依赖环境、资产根目录和新目录骨架。
- 校验全部必需 USD 及其依赖可打开。
- 验证机器人、工作站、目标和 prim 名称。
- 验收标准：单环境静态场景加载成功，无缺失资产或错误 prim 契约。

### 阶段 B：场景和任务真值

- 迁移场景位姿、机器人初态、TCP 标定和目标状态。
- 迁移 TCP、姿态和停车状态机纯逻辑测试。
- 验收标准：相同输入在新旧工作区产生一致的 TCP 状态、姿态误差和成功判定。

### 阶段 C：Differential IK

- 实现六维任务空间动作、Jacobian 读取、阻尼最小二乘、关节限幅和关节位置目标写入。
- 先用固定命令和固定目标验证，不启动 RL。
- 验收标准：位置与姿态误差稳定下降，无 NaN、关节越界或持续奇异抖动。

### 阶段 D：基础 RL

- 接入观测、奖励、终止和 SB3 PPO。
- 固定最简单目标状态完成短程训练和回放。
- 验收标准：环境可并行 reset，奖励和终止统计有限且可解释，Checkpoint 可独立回放。

### 阶段 E：课程 RL

- 接入课程等级、晋级统计、训练分布和 Checkpoint 继承。
- 逐步扩展姿态、低速停留、全部目标状态和随机化。
- 验收标准：最终固定评估条件下覆盖全部目标状态，并输出分状态成功率、误差、碰撞率和完成时间。

## 8. 项目 Skill 迁移方案

### 8.1 当前项目 Skill 清单与结论

当前仓库 `.codex/skills/` 下共有 3 个项目级 Skill：

| Skill | 当前用途 | 迁移结论 | 原因 |
|---|---|---|---|
| `project-engineering-guardrails` | 约束所有项目文件修改、模块边界、编码、日志和验证 | 改造后优先迁移 | 规则与新工作区长期开发直接相关，但项目名称、目录和控制器边界需要更新 |
| `isaaclab-scene-debug` | 排查资产放置、拆分 USD、碰撞、相机和数值配置 | 改造后迁移 | 新方案复用同一资产和场景，仍需要这些排查流程；当前文件路径和 Lula 时代入口不适用于新目录 |
| `robot-dataset-governance` | 约束轨迹采集、schema、存储、导出和回放 | 首期不迁移，按条件启用 | 当前新方案目标是 Differential IK 与在线课程 RL，并未包含数据集生产；直接启用会引入无关规范和依赖 |

推荐迁移顺序：

1. 创建新工作区时先迁移工程规范 Skill。
2. 新场景骨架建立后迁移并改造 IsaacLab 场景调试 Skill。
3. 只有在明确加入轨迹采集、离线 RL、模仿学习或数据集导出后，才迁移数据集治理 Skill。

### 8.2 `project-engineering-guardrails`：改造后优先迁移

#### 应迁移内容

- 修改前检查 Git 状态和目标文件 diff，保留用户已有改动。
- 中文注释、UTF-8 和避免无关整文件格式变化的要求。
- 文件规模、职责拆分、公共接口稳定性和循环依赖检查。
- 四部分工程边界：配置、纯逻辑、Isaac 适配、生成数据。
- “配置 → 逻辑 → Isaac 适配/执行 → 数据输出”的单向依赖。
- 诊断日志必须可关闭，保留启动、资产、控制器和写入失败上下文。
- 修改后的 diff、语法、单元测试和 Isaac 运行验证要求。

#### 需要改写内容

- 将 frontmatter 中的“当前 IsaacLab AUBO 项目”改为新项目正式名称。
- 将控制器边界明确为 Differential IK 适配层，不再泛指现有 Lula 集成。
- 将新工作区目录写入规则，例如 `configs/`、`logic/`、`runtime/`、`tasks/tcp_docking/` 和 `tests/`。
- 增加课程配置约束：课程等级、晋级阈值、采样分布和最终评估阈值必须分离；最终任务真值不得随课程静默变化。
- 增加训练产物约束：Checkpoint、日志、配置快照和随机种子必须关联，但不得作为源码提交。
- 将验证命令替换为新工作区实际可执行的 lint、pytest、环境 smoke test 和 Isaac 启动命令。

#### 迁移方法

1. 复制 `.codex/skills/project-engineering-guardrails/` 到新工作区同名目录。
2. 修改 `SKILL.md` 的 frontmatter、目录约定、Differential IK 边界和验证命令。
3. 同步复制并更新 `agents/openai.yaml` 中的显示名称、简述和默认提示。
4. 检查 Skill 中提到的每个路径在新工作区真实存在，不保留旧项目文件名作为隐式兼容入口。
5. 用一次小型配置修改验证该 Skill 能触发修改前检查、最小改动、测试和交付报告流程。

### 8.3 `isaaclab-scene-debug`：按新架构改造后迁移

#### 应迁移内容

- 调试前先确认当前文件布局、Git 状态和配置事实来源。
- 将问题拆分为资产放置、物理/碰撞、运行时诊断三个层次。
- 资产根目录、工作站基准位姿、机器人局部位姿和目标局部位姿使用单一输入源。
- 世界位姿和相机最终位姿尽量由基准配置派生，而不是重复写常量。
- 拆分工作站 USD 只负责资产加载和语义分组，不在放置模块中重新生成碰撞 schema。
- 接触问题检查传感器激活、挂载名称、prim 表达式和运行时 contact tensor。
- CameraSensor 与 viewport camera 分开排查；训练默认可以关闭视觉传感器。
- 使用与改动风险匹配的最小验证，不在缺少 Isaac 环境时宣称仿真通过。

#### 需要删除或替换的旧内容

- 删除对 `configs/Testcfg.py`、`configs/RLcfg.py`、旧 `scripts/test.py` 等固定旧路径的依赖。
- 删除把 `WorkStation_All` 视为活动场景根路径的旧假设；新项目继续按 `/station/{static,interactive,dynamic}` 组织拆分资产。
- 将旧相机常量名称替换为新工作区 `configs/scene.py` 或独立 camera 配置中的实际名称。
- 不迁移 Lula URDF、Lula robot description 和 Lula 求解诊断流程。

#### 应新增的 Differential IK 调试内容

- 验证机械臂关节和 `Flange` body id 的解析结果、顺序和唯一性。
- 验证固定基座/浮动基座对应的 Jacobian body 索引规则。
- 同时记录策略六维原始动作、限幅后任务空间命令、Differential IK 关节增量和最终关节目标。
- 检查阻尼系数、Jacobian 条件数或奇异标志、关节限位投影及单步最大变化。
- 分别比较期望 TCP 位姿变化、实际 TCP 位姿变化和控制周期内的执行完成率。
- 检查 `sim.dt`、decimation 和动作保持方式，避免将控制频率问题误判为 IK 收敛问题。
- 为课程学习增加当前等级、目标分布、启用约束和晋级统计的诊断入口。

#### 迁移方法

1. 不直接覆盖复制旧 Skill；在新工作区创建 `.codex/skills/isaaclab-tcp-diffik-debug/`。
2. 以当前 `isaaclab-scene-debug/SKILL.md` 为基础，保留资产、放置、碰撞和相机工作流。
3. 将所有路径重写为第 5 节推荐的新工作区目录，并加入 Differential IK 与课程状态排查流程。
4. frontmatter description 应明确 TCP 停靠、Differential IK、课程状态和 AUBO 资产范围，使 Skill 能被相关任务准确触发。
5. 分别用资产缺失、错误 body 名称、ContactSensor 未挂载和 Jacobian 维度错误四类故障验证操作步骤是否可执行。

### 8.4 `robot-dataset-governance`：首期不迁移，按条件整体启用

#### 首期不迁移原因

- 当前 PRD 只定义在线课程 RL 训练和 TCP 停靠评估，没有定义轨迹数据集交付物。
- Skill 强制依赖 `docs/机器人轨迹数据集规范.md`，单独复制 `SKILL.md` 会产生缺失事实来源。
- 现有规范包含当前三维位置动作和 Lula 控制器目标的项目映射，与新六维 Differential IK 动作链不一致。
- 数据工具还会引入 Parquet、Zarr、序列化、校验和外部格式导出等额外依赖，不应进入控制方案的首期关键路径。

#### 满足以下任一条件时应启用

- 开始采集教师策略轨迹；
- 使用离线 RL、行为克隆或视觉模仿学习；
- 需要 RLDS、LeRobot 或 Robomimic 导出；
- 需要保存多频率相机、接触或控制器诊断数据；
- 需要长期可追溯地比较不同课程、控制器或 Checkpoint。

#### 条件启用时必须整体迁移的内容

- `.codex/skills/robot-dataset-governance/SKILL.md`；
- `.codex/skills/robot-dataset-governance/agents/openai.yaml`；
- `docs/机器人轨迹数据集规范.md`；
- `configs/dataset_cfg.py`；
- `tools/dataset/` 下的 schema、writer、builder、extractor、validator 和 Isaac recorder；
- `scripts/collect_dataset.py`；
- 数据逻辑和往返序列化测试。

#### Differential IK 项目映射需要更新的内容

- 将动作 representation 从三维 `delta_position` 更新为六维增量位姿，并明确旋转表示、坐标系、单位和归一化方式。
- 同时记录 `policy_raw`、限幅后的 TCP 命令、Differential IK 关节目标和实际执行结果。
- 在动作诊断中记录位置/旋转裁剪、阻尼、奇异状态、关节限位投影和控制器饱和信息。
- 将 `curriculum_level`、课程配置版本、目标采样分布和晋级原因写入 Episode 或训练上下文。
- 保持 `terminated`、`truncated`、`success` 和 `invalid` 分离，不能用课程阶段成功替代最终停靠成功。
- 更新规范中的“当前 AUBO 项目映射”，并在启用前决定 schema 版本和旧数据兼容策略。

### 8.5 Skill 迁移交付清单

新工作区首期应形成：

```text
.codex/skills/
├─ project-engineering-guardrails/
│  ├─ SKILL.md
│  └─ agents/openai.yaml
└─ isaaclab-tcp-diffik-debug/
   └─ SKILL.md
```

首期不创建空壳 `robot-dataset-governance`。数据集需求确认后，再将该 Skill、规范、数据工具和测试作为一个完整变更引入。

迁移验收要求：

- 每个 Skill 的名称、description 和触发范围与新项目一致；
- Skill 引用的文件和命令全部能在新工作区解析；
- 不包含旧项目绝对路径、旧任务名或 Lula 专用运行依赖；
- 工程规范 Skill 与专业 Skill 的职责不重复，专业 Skill 明确要求同时遵守工程规范；
- 至少使用一个代表性任务验证每个已迁移 Skill 的触发、工作流和交付内容。

## 9. 粗粒度 PRD

### 9.1 方案目的

在新的独立工作区中建立一条面向 AUBO 带夹爪 TCP 停靠的技术路线：使用 Differential IK 承担任务空间到关节空间的实时控制，使用课程强化学习逐步训练策略完成接近、姿态调整、低速进入和稳定停靠。

最终目标与当前项目保持一致：TCP 在全部配置目标状态下到达指定停靠位姿，并同时满足位置、姿态、速度、连续停留和安全约束。新方案应降低端到端策略直接学习关节控制的难度，并消除对 Lula 非线性 IK 求解链的运行依赖。

### 9.2 方案实现方法

- 以独立任务包实现新方案，迁移当前项目的资产契约、场景位姿、TCP 标定、目标状态、最终成功判据和安全约束；保留当前 Lula 方案作为对照基线。
- 策略输出机器人根坐标系下的六维 TCP 增量命令，包括三维位置增量和三维旋转增量。
- Differential IK 每个物理控制周期读取末端 Jacobian 和当前关节状态，使用阻尼最小二乘计算关节增量，并经过关节位置、关节速度、单步变化和奇异性保护后生成关节位置目标。
- 策略观测包含关节位置与速度、TCP 到 preposition 的位置误差、TCP 与目标的姿态误差、TCP 线速度与角速度、目标状态以及当前课程等级。
- 课程训练按难度逐级展开：固定目标的位置接近；加入姿态对齐；加入低速进入和连续停留；扩展到全部目标状态、碰撞约束和随机化。
- 使用滚动成功率、有效 episode 数和安全失败率决定课程晋级；下一等级从上一等级 Checkpoint 初始化。课程状态和晋级原因写入训练日志及配置快照。
- 课程可以改变目标采样范围、初始距离、辅助奖励和约束启用顺序，但最终成功状态机及正式评估阈值保持不变。
- 最终评估按目标状态分别统计停靠成功率、位置误差、姿态误差、TCP 速度、碰撞率、超时率和平均完成时间，并与当前 Lula 基线使用同一场景和终止定义进行比较。
