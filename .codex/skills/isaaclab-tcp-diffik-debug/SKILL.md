---
name: isaaclab-tcp-diffik-debug
description: 诊断 CurriculumRL 中 AUBO 带夹爪 TCP 停靠的 Isaac Lab 场景、拆分 USD 资产、坐标变换、接触传感器、Differential IK、Jacobian、控制频率及课程状态问题。遇到资产缺失或错位、Flange/body/关节解析错误、碰撞与 ContactSensor 异常、TCP 位姿或速度错误、IK 不收敛/抖动/越界、课程晋级或正式评估异常时使用。
---

# AUBO TCP Differential IK 调试

## 先应用工程规范

在诊断或修改前，完整读取并遵守 `.codex/skills/project-engineering-guardrails/SKILL.md`。本 Skill 只增加场景和控制领域的诊断流程，不替代 Git、架构、编码和交付规则。

## 建立诊断基线

0. 在当前工作站优先使用已验证的 Isaac Lab 解释器 `D:\Anaconda\envs\isaaclab\python.exe` 直接运行仓库脚本，例如 `D:\Anaconda\envs\isaaclab\python.exe scripts/list_envs.py`。先核验该文件存在且 `import isaaclab` 成功；若失效，报告环境故障，不静默切换到普通 Python 或仅含 Isaac Sim 的启动器。
1. 读取 `docs/DifferentialIK课程RL_TCP停靠方案与项目迁移说明.md` 中与故障相关的契约。
2. 运行 `git status --short`，检查目标文件 diff，并用 `rg --files source scripts docs` 确认当前实现位置。
3. 记录复现命令、Isaac Sim/Isaac Lab/Python 版本、设备、环境数、随机种子、`sim.dt`、decimation、任务名和课程等级。
4. 先把问题归入资产放置、物理/碰撞或运行时控制；证据跨层后再扩大范围。
5. 缺少 Isaac 环境时只完成静态契约和纯逻辑验证，不宣称仿真通过。

## 核对固定契约

以迁移说明为准核对，不在多个模块重复常量：

- 受控实体为 `AUBObot`，第二机器人为 `AUBObot_2`，articulation prim 为 `AUBO_E5`。
- 末端刚体为 `Flange`；机械臂关节为 `Joint1` 至 `Joint5` 与 `Flange`；`UpperFinger`、`DownFinger` 不进入机械臂 IK 向量。
- 停靠目标 scene key 为 `ws_interactive_reagent_01_sample_bottle`。
- 坐标系为世界 `W`、环境 `E`、机器人根 `B`、法兰 `F` 和 TCP `T`；四元数统一使用 `wxyz`。
- TCP 法兰局部平移为 `(0.0, -0.12, 0.102) m`。将它视为可替换标定参数，不写入 IK 或奖励实现内部。
- 最终停靠进入/退出距离为 `0.04/0.055 m`，TCP 速度阈值为 `0.03 m/s`，姿态阈值为 `10°`，连续停留为 `2` 个控制步。
- 仿真频率为 `120 Hz`，decimation 为 `30`，策略频率为 `4 Hz`。先验证动作保持语义，再判断 IK 收敛。

## 第一层：资产与放置

1. 查找唯一资产根配置；拒绝散落的绝对路径和仅复制顶层 USD 的做法。
2. 验证机器人、工作站拆分资产、目标 USD 及其材质、纹理和子 USD 依赖均可解析。
3. 验证工作站仍按 `/station/{static,interactive,dynamic}` 语义分组，不把 `WorkStation_All` 当作活动场景根。
4. 验证工作站基准位姿、机器人局部位姿和目标局部位姿从单一配置派生；逐环境世界位置必须正确加入 `env_origins`。
5. 分别记录配置局部位姿、派生世界位姿和仿真实际位姿。不要只凭 viewport 外观判断坐标正确。
6. 相机问题单独区分 CameraSensor 与 viewport camera；非视觉训练默认可关闭 CameraSensor。

资产缺失时输出缺失文件、引用它的 USD/配置、解析后的完整路径和资产根来源。不要用修改 prim 名称或关闭碰撞掩盖缺失依赖。

## 第二层：物理、碰撞与接触

1. 确认机器人 spawn 启用 `activate_contact_sensors=True`。
2. 确认 ContactSensor 表达式匹配 `{ENV_REGEX_NS}/AUBObot/AUBO_E5/.*`，并在多环境下解析到预期刚体。
3. 检查 contact tensor 的环境维、body 维、单位、刷新周期和聚合方式。
4. 仅从非法接触判定中排除 `Base_Link`；不要关闭整个机器人或工作站碰撞。
5. 验证拆分 USD 自带的 CollisionAPI 在新资产路径下仍有效。放置配置只负责加载与分组，不重新生成碰撞 schema。
6. 临时禁用碰撞时只针对明确 prim，并在交付中记录；诊断后恢复正式约束。

ContactSensor 未挂载时，依次报告 spawn 激活、prim 表达式解析、实际匹配 body、tensor 形状和更新时间，定位失败发生在哪一步。

## 第三层：Differential IK 运行链

在场景初始化后通过 `SceneEntityCfg.resolve()` 或等价机制解析实体，不依赖字符串列表的假定顺序：

1. 验证六个机械臂关节各唯一匹配，顺序与关节状态及 Jacobian 列一致。
2. 验证 `Flange` body 唯一匹配并可读取位姿、线速度、角速度和 Jacobian。
3. 明确 articulation 是固定基座还是浮动基座，并验证对应 Jacobian body 索引偏移。
4. 记录动作链各阶段：策略六维原始动作、缩放/限幅后 `B` 系 TCP 增量、期望 TCP 位姿、Jacobian 形状、阻尼或奇异标志、关节增量、限位投影和最终关节位置目标。
5. 检查位置与旋转裁剪、阻尼系数、Jacobian 条件数或奇异判据、关节位置/速度限制和单步最大变化。
6. 比较期望 TCP 变化、实际 TCP 变化和一个控制周期内的执行完成率；同时检查 `sim.dt`、decimation 与动作保持。
7. 验证 TCP 速度包含法兰角速度与局部偏置的叉乘项，并在 `W/E/B/F/T` 转换中保持单位和四元数顺序一致。

Jacobian 维度错误时，至少列出解析到的 joint/body 名称及索引、原始 Jacobian 形状、选取后的形状、固定/浮动基座假设和预期六关节形状。不要通过截断未知列临时通过。

## 核对停靠状态与课程状态

- 使用带进入/退出滞回的统一状态机，并保证同一控制步只提交一次状态更新。
- 分开记录距离、速度、姿态、dwell、非法接触、目标位移/速度和工作空间判定。
- Reset 必须按 `env_ids` 原子更新目标与缓存，并清空进展、里程碑、姿态、停靠区和 dwell 历史。
- 记录课程等级、目标采样分布、已启用约束、有效 episode 数、滚动成功率、安全失败率和晋级/回退原因。
- 始终用固定正式阈值单独评估；课程成功不能替代最终停靠成功。

## 用代表性故障验证流程

在不破坏正式资产的前提下，用最小测试或临时配置分别验证：

1. 缺失资产能在启动阶段给出明确路径和来源；
2. 错误 `Flange`/body 名称能在实体解析阶段失败；
3. 未挂载 ContactSensor 能在读取 contact tensor 前失败；
4. 错误 Jacobian 维度能在求解前失败并打印解析契约。

每类故障都应快速失败、指向具体层次，并在修复后用相同入口复验。不得保留故障注入改动。

## 输出诊断结论

交付根因或当前最窄故障边界、支持证据、修改内容、实际运行的验证、未运行项和剩余风险。区分“静态契约成立”“纯逻辑测试通过”“Isaac 场景启动通过”和“控制闭环行为通过”。
