# 阶段 C 施工记录

## C1 六维动作与纯逻辑求解

阶段/批次：C1 六维动作与纯逻辑求解  
状态：已验收
事实来源：`docs/DifferentialIK课程RL_TCP停靠方案与项目迁移说明.md`、`docs/多阶段施工参照.md`  
Git 基线：`28051f0`；施工前工作区无未提交改动。  

已实现：

- `configs/differential_ik.py` 集中定义根坐标系 B 下六维动作、平移/旋转尺度、DLS 阻尼、奇异阈值、单步变化、速度和位置 margin。
- `logic/differential_ik.py` 实现旋转向量与 `wxyz` 四元数转换、增量目标、六维位姿误差、TCP 偏置 Jacobian、阻尼最小二乘和三层关节目标投影。
- 策略动作先裁剪至 `[-1,1]`；单策略步最大平移为 `0.02 m`，最大旋转为 `5°`。
- DLS 使用 `Jᵀ(JJᵀ+λ²I)⁻¹e`；最小奇异值低于阈值时提高阻尼。
- 关节目标依次受单物理步变化、关节速度和资产实际位置限位约束；非有限输入在写入前失败。

纯逻辑测试覆盖零/π 旋转向量、六维动作缩放、TCP 偏置切向 Jacobian、满秩/奇异 DLS、单步/速度/位置保护以及错误 Jacobian 维度快速失败。

## C2 Isaac 动作适配链

阶段/批次：C2 Isaac 动作适配链  
状态：已验收

`tasks/tcp_docking/actions.py` 已实现：

1. 每个策略步读取当前 TCP，并生成一次根坐标系 B 下增量位姿目标；
2. 之后 `30` 个物理步持续跟踪同一目标，不重复累加策略增量；
3. 按资产解析顺序固定使用 `Joint1` 至 `Joint5` 和 `Flange` 六列；
4. 固定基座将 `Flange body_id=6` 映射为 `jacobian_body_id=5`；浮动基座路径显式增加 6 个根自由度列偏移；
5. 将 PhysX 世界系法兰 Jacobian 转到 B 系，并加入 TCP 偏置的 `ω×r` 线速度项；
6. 求解、保护后只向六个机械臂关节写位置目标，夹爪索引不进入求解；
7. 可选诊断保留原始/缩放动作、目标/当前 TCP、误差、最小奇异值、阻尼、保护标志、关节增量和最终目标。

未导入 Lula URDF、robot description、Lula 求解器或旧 `AuboTaskSpaceIKAction`。

## C3 V3 动态验证

阶段/批次：C3 V3 动态验证  
状态：已验收

执行入口：

```powershell
& 'D:\Anaconda\envs\isaaclab\python.exe' scripts\smoke_stage_c.py --headless --num-envs 4 --traceback-timeout-s 0
```

运行契约：Isaac Sim `5.1`、Isaac Lab `0.54.3`、PyTorch `2.7.0+cu128`、CUDA 设备、`4` 环境、`dt=1/120 s`、decimation `30`。

实测结果：

| 验证 | 结果 |
|---|---|
| Jacobian 契约 | 原始 `(4,9,6,8)`；固定基座；机械臂列 `0..5`；Flange body `6` / Jacobian body `5` |
| 零动作保持 | 30 物理步最大 TCP 平移漂移约 `1.1 mm`，小于 `2 mm` 门限；旋转小于 `0.01 rad` |
| 六轴方向 | `±x/±y/±z/±rx/±ry/±rz` 全部方向正确；最小有效位移/转角均超过脚本门限 |
| 固定位置目标 | 六维误差范数约 `0.02049 -> 0.00450`，通过 `<0.006` 成功门限 |
| 固定六维目标 | 六维误差范数约 `0.04858 -> 0.00238`，通过 `<0.005` 成功门限 |
| 四目标/四环境 | 到对应 preposition 距离分别由 `[0.8053,0.6314,0.6003,0.5925] m` 降至 `[0.7768,0.5906,0.5623,0.5627] m` |
| 保护样例 | 人为 `+0.5 m` 大误差在四环境全部触发保护；应用关节增量未超过单步上限 |
| 奇异样例 | 纯逻辑病态 Jacobian 触发高阻尼，输出有限 |

V3 脚本同时逐物理步检查关节位置、速度和有限值，并在方向、收敛、环境隔离或保护任一门禁失败时立即终止。

其余验证：

```powershell
& 'D:\Anaconda\envs\isaaclab\python.exe' -m pytest tests -q
python -m compileall -q source\CurriculumRL\CurriculumRL scripts
python -m ruff check <阶段 A/B/C 目标 Python 路径> tests
python -m ruff format --check <阶段 A/B/C 目标 Python 路径> tests
git diff --check
```

结果：全部测试 `21 passed`；目标路径编译、Ruff check、Ruff format 和 diff 检查均通过。仓库未改动模板中原有的全量 Ruff 基线问题未纳入本阶段批量修复。

## 保持不变与剩余风险

- TCP 标定、四目标状态、最终停车状态机、安全阈值、执行器增益和 `120 Hz / 30 / 4 Hz` 频率契约均未改变。
- TCP 平移偏置 `(0.0, -0.12, 0.102) m` 仍是第一版近似值；本阶段已在 Jacobian 中正确使用该配置，但正式评估前仍需实测复核。
- 当前资产实测为固定基座；浮动基座索引分支按 Isaac Lab 官方规则实现，但未用本项目浮动基座资产动态验证。
- Isaac Sim 仍报告用户配置不可写及 KVDB 被其他 Kit 进程占用；本次控制验证与正常退出未受影响。
- 阶段 C 不注册 RL 环境、不实现奖励、不启动 PPO，也未调整执行器增益。

阶段 C 结论：C1、C2、C3 施工和 V3 动态验证已完成；用户于 2026-07-08 指示开始阶段 D，视为阶段 C 已验收。
