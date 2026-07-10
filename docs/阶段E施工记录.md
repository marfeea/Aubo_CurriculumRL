# 阶段 E 施工记录

阶段/批次：E-1 课程控制与固定条件评估入口  
状态：施工中，待 Isaac 动态验收  
事实来源：`docs/多阶段施工参照.md` 第 11 节、`docs/项目理解.md` 的“阶段 E”边界及阶段 D 现有任务代码。  

## 本批变更

- `configs/curriculum.py` 定义带版本的五级课程表、四目标状态采样概率、辅助奖励缩放及固定长度滚动窗口规则；没有复制 `configs/task.py` 的停靠/安全阈值。
- `logic/curriculum_state.py` 实现按 episode 归属等级分别累积的滚动窗口、批量提交、冷却期、升级/回退记录和 JSON 无损恢复。安全失败与停车成功同一步发生时按安全失败处理。
- 课程 reset 先结算本批旧 episode、至多切换一次全局等级，再为该批环境按新等级采样目标；每个环境保存自己的 episode 等级，避免异步 reset 把旧结果混入新等级窗口。
- 辅助奖励按当前 episode 等级缩放；最终成功奖励和安全失败惩罚不受课程缩放影响。L4 仍没有已批准的物理随机化，配置中明确为空。
- `scripts/sb3/train.py` 为每个 checkpoint 关联 `VecNormalize` 和课程状态 JSON；恢复阶段 E 训练时缺少任一配套状态会快速失败。
- `scripts/evaluate_stage_e.py` 以冻结 PPO/`VecNormalize`、关闭课程切换、固定 L4 课程观测和逐状态固定目标的方式，按多个种子输出成功率、最终误差、成功速度、碰撞/扰动/超时率与完成时间。

## 保持不变的任务契约

- TCP 标定、四个目标状态定义、停车进入/退出距离、姿态与速度限制、dwell、工作空间和安全阈值仍只来自 `configs/task.py`。
- 观测仍保留四维目标 one-hot 加一维课程等级；课程等级改为 `[0, 1]` 归一化值，L0 仍为 0，因此不改变阶段 D checkpoint 的观测形状或 L0 数值。
- 目标保持运动学刚体；在资产和接触行为完成验证前，不以“L4 完整随机化”名义启用未定义的物理扰动。

## 验证计划

```powershell
python -m compileall source/CurriculumRL/CurriculumRL scripts tests
python -m pytest tests/test_stage_b_logic.py tests/test_stage_d_logic.py tests/test_stage_e_logic.py
python -m ruff check source scripts tests
python -m ruff format --check source scripts tests
<ISAACLAB_ROOT>\isaaclab.bat -p scripts/list_envs.py
<ISAACLAB_ROOT>\isaaclab.bat -p scripts/smoke_stage_d.py --num-envs 4
<ISAACLAB_ROOT>\isaaclab.bat -p scripts/sb3/train.py --task CurriculumRL-TcpDocking-v0 --num_envs 4 --max_iterations 2 --rollout-steps 8 --run-tag stage-e-smoke
<ISAACLAB_ROOT>\isaaclab.bat -p scripts/evaluate_stage_e.py --checkpoint <CHECKPOINT> --num-envs 4 --episodes-per-state 4 --seeds 7
```

## 遗留风险与准入结论

- 当前晋级/回退数值标记为 `stage-e-v1-provisional`：必须用阶段 D 的足量零动作、随机动作与 PPO 基线复核后才能用于长训练。
- 当前机器只有 Isaac Sim Python，未配置 Isaac Lab 扩展与 `isaaclab.bat` 启动器；`C:\isaac-sim\python.bat scripts/list_envs.py` 已因 `ModuleNotFoundError: isaaclab` 停止，因此尚未执行 Isaac smoke、训练或正式评估。
- 仍需在 Isaac 中确认 reset 回调读取的是终局步的终止张量，且各目标状态的完成缓存与 SB3 auto-reset 同步。
- `sample_bottle` 运动学目标不会触发真实目标扰动；目标扰动安全率在最终验收前需要经过可扰动资产/接触场景验证。
- 阶段 D 尚未具备足量成功策略，本批不宣称课程有效、训练收敛或阶段 E 最终验收通过。

下一阶段准入结论：允许进入阶段 E 的静态与 smoke 验证；不允许标记项目迁移完成。  
