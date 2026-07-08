# 阶段 B 施工记录

## B1 纯逻辑迁移

阶段/批次：B1 纯逻辑迁移  
状态：已验收

事实来源：`docs/DifferentialIK课程RL_TCP停靠方案与项目迁移说明.md`、`docs/多阶段施工参照.md`  
工作区边界：保留阶段 A 全部未提交成果，不修改旧模板任务，不接入动作、奖励或训练。  

已实现：

- `logic/tcp_kinematics.py`：`wxyz` 四元数、法兰局部 TCP 偏置、`omega × r` 切向速度、世界系到机器人根坐标系变换。
- `logic/orientation.py`：目标姿态到期望法兰姿态、工具轴夹角和余弦对齐分数。
- `logic/parking_state.py`：进入/退出迟滞、速度、姿态、连续停留与同一控制步幂等提交。
- `logic/curriculum_state.py`：课程 episode 统计及晋级/回退状态转换；未接入训练。
- `logic/terminations.py`：工作空间、非法接触、目标位移/速度和超时判定。

V1 定向测试覆盖单/多环境、环境原点平移、四元数顺序、偏置切向速度、迟滞边界、dwell 清零、重复控制步、接触维度和环境隔离。代表性输入的预期结果由迁移说明中的固定契约锁定；旧工作区源码不在当前可访问项目树中，因此未执行旧模块与新模块的同进程逐函数对拍。

## B2 目标与 reset 数据流

阶段/批次：B2 目标与 reset 数据流  
状态：已验收

已实现：

- `logic/reset_state.py` 使用逐环境张量缓存目标状态、世界位姿、回合基准、preposition、期望法兰姿态及全部历史状态。
- `tasks/tcp_docking/events.py` 仅负责 Isaac 目标刚体位姿/零速度写入和缓冲回读；纯逻辑不导入 Isaac。
- `tasks/tcp_docking/dynamic_scene_cfg.py` 将样品瓶绑定为 `RigidObject`，供目标写入、速度清零和回读。
- 全环境 reset 与单环境部分 reset 均已在双环境仿真中执行；未选环境的目标状态与全部缓存保持不变。

reset 隔离检查位于写入/回读完成后、下一物理步之前。目标进入下一物理步后的重力或接触位移属于任务动态，由目标位移和速度失败条件判定，不属于 reset 写入误差。

## B3 接触、失败条件和最终状态机

阶段/批次：B3 接触、失败条件和最终状态机  
状态：已验收

已实现：

- ContactSensor prim 表达式继续只从 `configs/assets.py` 的 `{ENV_REGEX_NS}/AUBObot/AUBO_E5/.*` 读取。
- 双环境实测接触张量为 `(2, 7, 3)`，body 为 `Base_Link`、`Link_01` 至 `Link_05`、`Flange`。
- 非法接触只排除 `Base_Link`，其余刚体按逐环境最大净接触力判定，不跨环境聚合。
- 最终成功唯一来自 `parking_state.py` 的统一状态机；阶段 B 未实现奖励。
- 阶段 A articulation 参数适配器增加显式环境数校验，默认仍为单环境；双环境要求解析 4 台机器人，不放宽 prim 契约。

## 验证结果

执行命令：

```powershell
& 'D:\Anaconda\envs\isaaclab\python.exe' -m pytest tests\test_stage_b_logic.py -q
& 'D:\Anaconda\envs\isaaclab\python.exe' scripts\smoke_stage_b.py --headless --num-envs 2 --traceback-timeout-s 30
python -m compileall -q source\CurriculumRL\CurriculumRL scripts
python -m ruff check <阶段 A/B 新增或修改的 Python 路径> tests
python -m ruff format --check <阶段 A/B 新增或修改的 Python 路径> tests
git diff --check
```

当前结果：

| 验证 | 结果 |
|---|---|
| 阶段 B 纯逻辑 pytest | `10 passed` |
| 当前全部 tests | `14 passed` |
| 双环境 Isaac 动态 smoke | 通过；目标 `(2, 13)`，接触 `(2, 7, 3)` |
| 目标全量/部分 reset | 通过；未选环境无串扰 |
| ContactSensor body 解析 | 通过；7 个刚体，`Base_Link` 唯一 |
| 阶段 A/B 目标路径 Ruff check/format | 通过 |
| 仓库全量 Ruff | 未通过；22 项均位于未改动的 Isaac Lab 模板文件，未做无关批量格式化 |

## 保持不变与剩余风险

- 四个目标状态、TCP 标定、最终停车阈值、安全阈值、`120 Hz / 30 / 4 Hz` 和 `40 s` episode 均未改变；未引入 Lula 运行依赖。
- TCP 平移偏置 `(0.0, -0.12, 0.102) m` 仍是第一版近似值，正式评估前必须实测复核。
- Isaac Sim 仍报告用户配置不可写和另一个 Kit 进程持有 KVDB 锁；本次仿真创建、张量检查和正常退出未受影响。
- 旧工作区实现不可访问，当前一致性证据是迁移说明固定契约与代表性 golden 测试，不宣称已完成旧源码逐函数对拍。

阶段 B 结论：B1、B2、B3 的 V1/V2 已通过，用户于 2026-07-08 确认验收，允许进入阶段 C。
