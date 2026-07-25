# 阶段 E 施工记录

阶段/批次：E-1 课程控制与固定条件评估入口  
状态：施工中；Isaac smoke 与恢复训练通过，正式逐状态评估阻塞
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

## 2026-07-15 至 2026-07-16 动态验收执行

已验证本机 Isaac Lab 解释器为 `D:\Anaconda\envs\isaaclab\python.exe`：Python 3.11.15、Isaac Lab 0.54.3、Isaac Sim 5.1.0.0、PyTorch 2.7.0+cu128，CUDA 12.8 可用，设备为 NVIDIA GeForce RTX 4060。以下命令均直接使用该解释器：

- `scripts/list_envs.py --keyword TcpDocking` 正常注册 `CurriculumRL-TcpDocking-v0`。
- `python -m compileall -q source/CurriculumRL/CurriculumRL scripts tests` 通过；`python -m pytest -q` 为 28 项通过。环境未安装 Ruff，因此未执行 Ruff 门禁。
- `scripts/smoke_stage_d.py --num-envs 4 --steps 8 --headless` 通过：观测形状 `(4, 29)`、动作形状 `(6,)`，部分 reset 未影响其他环境，8 步零动作/随机动作的观测、奖励与终止张量均为有限值；奖励范围为 `[-0.074487, 0.112148]`，未触发停车成功、越界、非法接触、目标扰动或超时。
- `scripts/sb3/train.py --task CurriculumRL-TcpDocking-v0 --num_envs 4 --max_iterations 2 --rollout-steps 8 --run-tag stage-e-smoke --headless` 通过，共执行 64 timestep，并生成 PPO、`VecNormalize` 与课程 JSON 三件套。
- 使用上述 checkpoint 执行 `--max_iterations 1 --rollout-steps 2 --run-tag stage-e-resume-smoke` 的恢复训练通过；已保存模型与当前 Python、SB3、PyTorch、NumPy 和 Gymnasium 版本一致，课程状态与归一化状态均成功恢复并再次保存。
- `scripts/evaluate_stage_e.py --checkpoint <smoke-checkpoint> --num-envs 4 --episodes-per-state 4 --seeds 7 --headless` 未完成：第一个目标状态的 episode 批完成后进入第二个目标状态，但脚本在同一 Python/SimulationApp 内关闭并重建第二个 Isaac 环境后持续约 24 分钟不再完成 episode，仍持续占用 CPU 和约 3.2 GB 内存，最终人工终止。由于脚本只在四个状态全部完成后打印 JSON，本次没有形成可交付的逐状态指标或 V5 报告。

Isaac 启动期间反复出现 Isaac Sim `user.config.json` 无法保存以及 KV 数据库被其他 Kit 进程锁定的警告；注册、smoke、初始训练和恢复训练均正常退出，因此当前不把这些警告判定为上述正式评估阻塞的根因。当前最窄故障边界是正式评估入口在单一 SimulationApp 中顺序销毁并重建多个 Manager-Based 环境的运行链，需要单独诊断后用相同入口复验。

## 遗留风险与准入结论

- 当前晋级/回退数值标记为 `stage-e-v1-provisional`：必须用阶段 D 的足量零动作、随机动作与 PPO 基线复核后才能用于长训练。
- 本机已确认可使用 `D:\Anaconda\envs\isaaclab\python.exe` 启动 Isaac Lab；环境注册、动态 smoke、极短训练和恢复训练已经通过。正式逐状态评估仍受上述第二次环境重建阻塞，不能据此宣称阶段 E 已验收。
- 仍需在 Isaac 中确认 reset 回调读取的是终局步的终止张量，且各目标状态的完成缓存与 SB3 auto-reset 同步。
- `sample_bottle` 运动学目标不会触发真实目标扰动；目标扰动安全率在最终验收前需要经过可扰动资产/接触场景验证。
- 阶段 D 尚未具备足量成功策略，本批不宣称课程有效、训练收敛或阶段 E 最终验收通过。

下一阶段准入结论：允许进入阶段 E 的静态与 smoke 验证；不允许标记项目迁移完成。  
