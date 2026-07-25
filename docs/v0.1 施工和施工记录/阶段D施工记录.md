# 阶段 D 施工记录

## D1 L0 环境组装

阶段/批次：D1 固定目标基础 RL 环境  
状态：待验收  
事实来源：`docs/DifferentialIK课程RL_TCP停靠方案与项目迁移说明.md`、`docs/多阶段施工参照.md`  
Git 基线：`32b4a69`；施工前工作区无未提交改动。  

已实现：

- 注册正式任务 `CurriculumRL-TcpDocking-v0`，入口为 `ManagerBasedRLEnv`。
- 将阶段 C 的六维 Differential IK 包装为 ActionTerm，保持“策略步生成一次增量目标、30 个物理步持续跟踪”的控制契约。
- 策略观测为 29 维：六关节位置/速度、TCP 到 preposition 的三维位置误差、三维姿态误差、TCP 六维速度、四状态 one-hot 和 L0 课程等级。
- L0 固定使用 `sample_bottle_state_01`；目标设为运动学刚体，避免固定目标在重力下移动并错误触发 `target_disturbed`。阶段 E 才恢复可扰动目标分布。
- reset 同时恢复场景、目标、动作目标、停车状态和奖励历史；部分 reset 不更新其他环境缓存。
- `terminated` 包含停车成功、工作空间越界、非法接触和目标扰动；`time_out` 单独标记为 truncated。

## D2 奖励与统计

阶段/批次：D2 奖励状态与安全终止  
状态：待验收  

奖励包含距离进展、历史最短距离进展、邻近度、内区停靠质量、低速停车、首次进入、工具轴进展、最终成功和安全失败。最终成功只读取统一停车状态机；首次进入和历史最优只在新事件发生时给奖。所有跨步历史支持逐环境 reset，并通过控制步编号保证 termination、reward 和 observation 重复查询时不重复提交状态。

## D3 SB3 PPO 与 V4 动态验证

阶段/批次：D3 训练、Checkpoint 和独立回放链路  
状态：施工中  

训练入口新增运行标签和课程等级元数据。日志目录、Checkpoint 名称和配置快照关联任务、L0、seed 与 run tag；周期 Checkpoint 同步保存 VecNormalize，回放按跨平台路径规则解析对应归一化状态。`--rollout-steps` 和 `--steps` 仅用于有限时长 smoke，生产默认 PPO rollout 保持 32 步。

已执行验证：

```powershell
& 'D:\Anaconda\envs\isaaclab\python.exe' -m pytest tests -q
& 'D:\Anaconda\envs\isaaclab\python.exe' scripts\smoke_stage_d.py --headless --num-envs 4 --steps 8
& 'D:\Anaconda\envs\isaaclab\python.exe' scripts\sb3\train.py --task CurriculumRL-TcpDocking-v0 --headless --num_envs 4 --max_iterations 1 --rollout-steps 2 --seed 7 --run-tag smoke2
& 'D:\Anaconda\envs\isaaclab\python.exe' scripts\sb3\play.py --task CurriculumRL-TcpDocking-v0 --headless --num_envs 1 --steps 1 --checkpoint <SMOKE_CHECKPOINT>
```

结果：

- 全量纯逻辑测试 `24 passed`。
- V4 通过 4 环境全量/部分 reset、29 维观测、6 维动作和 8 步零/随机动作检查；观测与奖励有限，奖励范围约 `[-0.056906, 0.129638]`，无意外终止或超时。
- 8 样本 PPO smoke 完成，保存带任务/L0/seed/tag 的模型及 VecNormalize；独立进程成功加载二者并执行推理步。
- 新增 `scripts/evaluate_stage_d.py`：在不改变 L0 任务真值的前提下，以相同任务、seed、并行环境数和完整 episode 长度统计零动作、随机动作或 PPO 的回报、成功、安全失败和各终止原因；PPO 评估必须加载与 checkpoint 配套的 VecNormalize。
- 2026-07-10 使用 4 环境、seed 7、每策略 4 个完整 episode 对评估入口做 V4 接线验证：零动作全部在 160 个策略步超时（平均回报 `0.831989`、成功 `0/4`、安全失败 `0/4`）；随机动作平均回报 `-0.427876`，`3/4` 因 `outside_workspace` 终止、`1/4` 超时；既有 8 样本 PPO smoke checkpoint 能在独立进程加载其 VecNormalize，但 `4/4` 因 `outside_workspace` 终止（平均回报 `-0.490538`、成功 `0/4`）。这证明统计、终止归因和独立回放链路有效，不把 smoke 结果误作学习趋势。
- Isaac Sim 持续报告用户配置不可写和 KVDB 被其他 Kit 进程占用；不影响上述成功运行，但并发启动偶有资产初始化变慢。

未完成门禁：已形成可复现的零动作、随机动作和 PPO 同口径评估入口，但当前仅完成每策略 4 个 episode 的接线验证；尚未执行足以判断学习趋势的短程 PPO 训练，也没有成功率优于零动作和随机动作的已训练 checkpoint。因此阶段 D 总体保持“施工中”，不进入阶段 E。

保持不变与风险：TCP 标定、停车阈值、安全阈值、`120 Hz / 30 / 4 Hz` 和最终成功真值未改变。TCP 偏置仍需在正式评估前实测复核；L0 运动学固定目标不能替代阶段 E 的目标扰动安全验证。
