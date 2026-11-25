# pi05-memory-vla

在 π0.5（OpenPI）上加一个"记忆头"的 VLA：模型每 20 步除了输出动作，还写一段三行的文字记忆（动作、子目标清单、任务是否完成），下一轮再把这段记忆读回去。用来做需要记住"做到哪一步了"的长程任务。

本仓库是 2026-11-18 的代码快照，包含：仿真（RoboMemArena）上的训练和测评代码，以及把同一套方法搬到真机数据上的标注、训练配置和部署参考代码。

> 真机同学请直接看 [`docs/HANDOFF_REAL_ROBOT.md`](docs/HANDOFF_REAL_ROBOT.md)。

## 当前结果（仿真，RoboMemArena 26 个任务，每个任务 10 回合）

| 模型 | 步数 | 任务成功率 TSR | 阶段完成率 CSR |
|---|---|---|---|
| 本仓库 C_v3 | 1 万 | 28.5 | 45.3 |
| 本仓库 C_v3 | 2 万 | 39.6 | 61.7 |
| 上一版 C 组 | 5 万 | 38.5 | 58.2 |
| 榜单 | — | BATON 57.7 / SimpleMemVLA 63.6 / Astra 67.3 | — |

C_v3 计划训到 5 万步，3 万步以后的结果还没有出。测评规则：每 20 步重新规划，不用门控，记忆格式错不结束回合，不使用仿真器真值。

## 方法一句话版

- **输入**：三张 224×224 图（画了物体轨迹的主视角图、腕部图、干净的主视角图）、机器人状态、任务指令、上一轮的记忆文字、上一轮每个物体的隐向量。
- **输出**：未来 20 步动作；新的记忆文字；每个物体一个 64 维隐向量。
- **记忆文字**固定三行：

  ```
  Action:blue bowl: left -> picked | orange bowl: still
  Subgoal:done 3
  Task:Incomplete
  ```

  - `Action` 行：每个被跟踪物体一句，颜色就是轨迹图里的颜色；内容是当前子目标内累积的方向词和事件词，最多 12 个。
  - `Subgoal` 行：对清单的增量修改。第一轮写 `init [条目1; 条目2; ...]`，之后写 `none` 或 `done k`。清单本身由程序（Ledger Driver，`ledger_v2.py`）维护，模型只负责"打勾"。
  - `Task` 行：`Incomplete` 或 `Complete`。写出 `Complete` 后部署程序停止。
- **轨迹图**：用 SAM3 逐帧跟踪任务相关物体，把最近 21 帧的中心点连成线画在主视角图上。
- **方案 C**：除文字外，每个物体多一个隐向量，用对比学习得到的目标向量监督（`rma_latents.py`），外加一个动词分类头。

## 目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `code_causal_v3/` | **训练代码和策略服务**：`train.py`、`model_data.py`、`joint_model.py`、`policy.py`、`policy_ens.py`、`evaluate_openloop.py`、`memory_format.py`、`ledger_v2.py`、标注规则 `causal_labels.py`、隐向量 `rma_latents.py` | 7 个训练源文件（`train.py` 校验哈希的那 7 个）与正在跑的 C_v3 训练逐字节一致；配置文件里的个人账号名和 W&B 账号已替换为占位符 |
| `code_eval_v5/` | **仿真测评 / 部署适配**：`rma_queue_v5.py` → `rma_eval_v5.sh` → `rma_adapter_v5.py`，感知 `rma_perception_v5.py`，追踪 `rma_tracker_v5.py` | 当前测评用的版本 |
| `build/build_v2r/`、`build/build_v3/` | 仿真数据集的生成脚本和训练启动脚本（`train_C_v3.sh`） | 当前数据集就是它们生成的 |
| `tracker/` | SAM3 流式追踪封装 `rma_sam3_stream.py` 和仿真用的 v5 规则 | 需要另外准备 SAM3 源码和权重 |
| `renderer_driver/` | 轨迹图渲染器的 Python 端：`render_adapter.py`、`data_common.py`（JS 端在 `code_causal_v3/assets/`） | 需要 Playwright Chromium |
| `contracts/contract_C_v3.json` | 正在跑的 C_v3 训练的完整配置 | 路径是集群上的，换机器要改 |
| `scripts/` | 跨集群 checkpoint 视图 `mkview.py`、自动测评 `eval6_auto.sh`、成绩汇总 `agall.py`、视频打包 `pack6.py`、"勾多了"统计 `overtick.py`、仿真 runner 启动脚本 | 辅助 |
| `real_robot/` | **真机**：数据标注、数据集组装、检查、部署参考代码 | 新写的，见交接文档 |
| `docs/` | 交接文档 | |

`code_causal_v3/` 和 `code_eval_v5/` 里有很多历史实验文件（`closedloop_*`、`probe_*`、`rma_adapter.py`、`rma_adapter_v2.py`、旧的 `contract_*.json` 等），为了和集群上的目录完全一致没有删。当前流程只用到上表列出的文件。两个目录里同名的 `model_data.py`、`memory_format.py` 以 `code_causal_v3/` 的为准，测评时务必 `--code-dir code_causal_v3`。

## 仓库里没有、需要自己准备的东西

| 依赖 | 用途 | 说明 |
|---|---|---|
| OpenPI（我们用的是 `~/SANE` 这个 fork 的虚拟环境，jax 0.5.3） | 模型本体 | `contract.model.openpi_root`、`contract.model.python` |
| `pi05_base` 权重、`paligemma_tokenizer.model` | 初始化 | `contract.model.base_checkpoint`、`contract.model.tokenizer` |
| SAM3 源码（commit `f1647e8d`）和 `sam3.pt` | 追踪 | 放在 `TRACKER_SAM3_ROOT/src/` 下，另需一个带 torch 的环境 |
| Playwright + Chromium | 画轨迹图 | `PLAYWRIGHT_BROWSERS_PATH` |
| RoboMemArena 基准和它的 runner | 仅仿真测评 | https://github.com/OpenHelix-Team/RoboMemArena |

代码里的绝对路径（`xxxx/...`、`xxxx/...`、`xxxx/...`）是我们集群上的位置，换环境时用对应的环境变量或直接改。

## 训练

```bash
# 四个数字必须和 contract.training 里的完全一致，否则 train.py 拒绝启动
cd code_causal_v3
CUDA_VISIBLE_DEVICES=0,1,2,3 PI05_DEVICES=4 PI05_PER_DEVICE_BATCH=32 \
PI05_TOTAL_UPDATES=50000 PI05_CHECKPOINT_INTERVAL=10000 \
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONDONTWRITEBYTECODE=1 \
python -u train.py --contract <数据集>/contract_C_v3.json --dataset <数据集> --output <输出目录> --no-wandb
```

- `train.py` 是一个监督进程，每个 checkpoint 区间启动一个子进程；同一条命令再跑一次会从最新的 checkpoint 续训。
- 每个 checkpoint 存完后会自动做一次开环验证（需要 2 张卡），4 张 L20X 上约 2.5 小时，加上重启约 3 小时不走步，这是正常的。
- **训练期间不要改 `code_causal_v3/` 里的任何文件**：每次存盘都会重新校验 7 个源文件的哈希，变了就中止并丢弃这次存盘。要改代码请复制一份目录。
- 4×32 的 batch 下每张卡峰值约 34–38 GB 显存，L20X 上约 5.5 秒一步。

## 仿真测评

```bash
python code_eval_v5/rma_queue_v5.py <输出目录> "C_v3_020000=<checkpoint 目录>" \
  --tasks 1,4,8,10,20,25 --trials 10 --block 5 --seed 50 --gpus 0,1,2,3,4,5,6,7 --max-running 8 --free-mb 40000 \
  --code-dir code_causal_v3 --max-steps 2500 --replan 20
python scripts/agall.py <输出目录>        # 每个任务的 TSR / CSR
```

checkpoint 的 `metadata.json` 里记着训练机器上的 OpenPI 路径。在另一台机器上测评前先用 `scripts/mkview.py SRC DST` 生成一个"视图"（权重是软链接，只改这个路径）。

## 已知问题

- **抓空后记忆仍写"已抓到"**：6 个代表任务上，2 万步时失败回合的 60% 是记忆进度明显超前于实际进度，之后模型不会补救。这是目前成功率的主要瓶颈。
- 同一个随机种子重跑，结果可能不同（动作采样有随机性），10 回合的成绩有 ±1–2 个回合的波动。
- 真机部分只做到"数据能被训练代码正确读取、部署感知链路和训练数据一致"，**没有在真机上训练或运行过**，详见交接文档。
