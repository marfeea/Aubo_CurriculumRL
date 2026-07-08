# 阶段 A 施工记录

## S0 前置基线

阶段/批次：S0 前置基线  
状态：已验收  
事实来源：`docs/DifferentialIK课程RL_TCP停靠方案与项目迁移说明.md`、`docs/多阶段施工参照.md`  
工作区基线：施工前 `git status --short` 无输出；分支 `master`，基线提交 `3bfbb06`。  

资产来源与解析策略：

- 外部资产目录为 `D:\Project\S2R\Asset`，USD 不复制进仓库。
- 唯一配置入口为环境变量 `CURRICULUMRL_ASSET_ROOT`；未设置时允许从当前工作区相邻的 `Asset` 目录自动发现。
- 必需资产为带夹爪 AUBO、拆分工作站和 `Reagent_01` 样品瓶；实验室背景为可选资产，不进入阶段 A 静态场景。

运行约定：

- 预留任务注册 ID：`CurriculumRL-TcpDocking-v0`，阶段 D 注册前不得复用为模板任务别名。
- 配置快照格式：UTF-8 JSON。
- 运行标签格式：`<task>_<curriculum-level>_seed<seed>_<tag>`。
- `checkpoints/`、`data/`、`logs/`、`renders/` 已加入顶层忽略规则；`tests/` 不再被忽略。

依赖实测：

| 组件 | 结果 |
|---|---|
| Isaac Sim | `5.1` |
| Isaac Sim Python | `3.11.15` |
| PyTorch | `2.7.0+cu128` |
| Isaac Lab | `0.54.3` |
| Stable-Baselines3 | `2.8.0` |
| GPU | NVIDIA GeForce RTX 4060，驱动 `591.86` |

TCP 标定风险：`(0.0, -0.12, 0.102) m` 仍为第一版近似值，待实测复核。  
下一阶段准入结论：S0 允许阶段 A 验收。  

## A1 目录与配置骨架

阶段/批次：A1 目录与配置骨架  
状态：已验收  

已建立：

- `configs/`：资产、场景、任务、Differential IK 接口、课程和训练基线。
- `logic/`、`runtime/`、`tasks/tcp_docking/`：按单向依赖建立包边界。
- 资产、prim、关节、目标状态、TCP 标定和最终安全阈值均只有一个配置事实源。
- 包根目录在缺少 Isaac 时允许导入纯配置；Isaac 环境中仍保留任务和 UI 自动注册行为。

保持不变的任务契约：两台机器人命名、`AUBO_E5` articulation、`Flange`、六个机械臂关节、两个夹爪关节、四个目标状态、最终停车阈值、`120 Hz / 30 / 4 Hz` 和 `40 s` episode 均按迁移说明录入。未加入 Lula 运行依赖。  

## A2 资产和 articulation 检查

阶段/批次：A2 资产和 articulation 检查  
状态：已验收  

执行命令：

```powershell
python scripts/inspect_assets.py --filesystem-only --json
& 'C:\isaac-sim\python.bat' scripts/inspect_assets.py
```

结果摘要：

- 三项必需资产均存在且可由 `Usd.Stage.Open` 打开。
- 必需资产的递归 USD、材质和纹理依赖无未解析项。
- 机器人 USD 的 default prim 为 `/Root`，articulation 为 `/Root/AUBO_E5`。
- `Joint1` 至 `Joint5`、`Flange`、`UpperFinger`、`DownFinger` 均唯一匹配。
- `Flange` 刚体唯一匹配；机器人和工作站均存在 `PhysicsCollisionAPI`。
- 实验室可选背景存在内部纹理相对路径告警，但阶段 A 不加载该资产，不影响核心场景门禁；后续若启用背景必须单独修复或确认解析策略。

Isaac Sim 启动还报告安装目录缓存不可写和 RTX shader cache 初始化告警；资产检查进程退出码为 0，USD 契约检查通过，但这些安装级告警仍保留为环境风险。  

## A3 单环境静态场景

阶段/批次：A3 单环境静态场景  
状态：已验收  

已实现：

- 静态场景只加载工作站最小子集、两台 AUBO 和样品瓶目标，不加载相机、RTX 传感器或实验室背景。
- 机器人 USD 的 `/Root/AUBO_E5` 子 articulation 被显式绑定，避免把 default prim 错当 articulation。
- spawn 开启接触报告；运行时检查机械臂/夹爪索引隔离、`Flange` 位姿与速度、Jacobian 形状和接触报告 schema。
- 执行器基线保持迁移值；外部 USD 自带的自碰撞与求解迭代参数由运行时适配恢复为迁移基线。

执行命令：

```powershell
& 'D:\Anaconda\envs\isaaclab\python.exe' scripts/smoke_env.py --headless --num-envs 1 --traceback-timeout-s 30
```

结果：进程退出码为 0，单环境静态场景在约 6 秒内完成创建、首个物理步和清理。两台机器人均解析到机械臂关节 `Joint1` 至 `Joint5` 与 `Flange`、夹爪关节 `UpperFinger` 与 `DownFinger`，`Flange` body id 均为 6，Jacobian 形状均为 `(1, 9, 6, 8)`。接触报告 schema、有限位姿/速度和机械臂/夹爪索引隔离检查通过。  

修复记录：

- 显式创建 `/station/{static,interactive,dynamic}` 纯 Xform 语义层级，修复子资产生成前父 prim 不存在的问题。
- 阶段 A 不加载外部 `GroundPlaneCfg`，消除非核心地面 USD 引用导致的 `add_usd_reference()` 阻塞。
- 关闭前调用 `SimulationContext.clear_instance()` 解除 timeline stop 回调，保证进程正常退出。
- 增加逐步骤即时日志、异常 traceback 和超时全线程栈转储。

运行环境仍报告 Isaac Sim 用户配置文件不可写及另一个 Kit 进程持有 KVDB 锁；本次场景检查和进程退出未受影响，作为环境维护项保留。  

## 当前验证汇总

| 验证 | 结果 |
|---|---|
| `git diff --check` | 通过 |
| `python -m compileall source/CurriculumRL/CurriculumRL scripts` | 通过 |
| 新增/修改文件定向 Ruff check | 通过 |
| 新增/修改文件定向 Ruff format check | 通过 |
| `python -m pytest tests/test_asset_contract.py` | `4 passed` |
| Isaac Sim USD 完整检查 | 三项必需资产通过 |
| Isaac Lab 单环境 smoke | 通过；退出码 0 |

阶段 A 结论：A1、A2、A3 的 V0/V2 门禁均通过，阶段 A 已验收，允许进入阶段 B。  
