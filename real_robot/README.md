# real_robot：把真机数据标成训练代码能用的格式，并在真机上部署

面向数据集 [`wwccww/yx_bowls`](https://huggingface.co/datasets/wwccww/yx_bowls)（双臂 Dobot，任务"借空盘子交换两只碗"）。背景、已验证和未验证的内容见 [`../docs/HANDOFF_REAL_ROBOT.md`](../docs/HANDOFF_REAL_ROBOT.md)，这里只讲怎么跑。

## 文件

| 文件 | 作用 | 运行环境 |
|---|---|---|
| `real_spec.py` | **所有约定的唯一出处**：相机裁剪框、用哪 7 维动作、物体顺序和颜色、6 条子目标、阈值 | 任意 |
| `real_extract.py` | 第 0 步：读 LeRobot parquet，取状态/动作，用右夹爪开合切出 6 个子目标的边界 | pandas + pyarrow |
| `real_tracker.py` | 两只碗的追踪器（SAM3 + 三条身份修复规则），标注和部署共用；`--serve` 是部署时的子进程 | torch + SAM3，1 张卡 |
| `real_track.py` | 第 1 步：解码视频、裁剪、逐帧跟踪 | 同上，另需 ffmpeg |
| `real_qa.py` | 轨迹质检：每条示范结束时两只碗是否真的互换了位置 | numpy |
| `real_build.py` | 第 2 步：画轨迹图，生成三张模型输入图、逐帧记忆标注 | Playwright Chromium + `renderer_driver/` |
| `real_assemble.py` | 第 3 步：划分训练/验证、归一化统计、清单、训练配置 | numpy |
| （`code_causal_v3/rma_latents.py`） | 第 4 步：方案 C 的隐向量目标，原样复用 | 训练环境 |
| `real_check.py` | 第 5 步：用**训练代码自己的加载器**把数据集读一遍并做一致性检查 | 训练环境（CPU 即可） |
| `real_deploy.py` | 部署主循环，带两种离线自检模式 | 见文件头 |
| `overlay_v2r.py` | 轨迹历史 → 渲染器输入（与仿真共用，原样拷贝） | — |
| `run_track.sh`、`run_build.sh` | 把上面几步串起来 | bash |

## 从原始数据到可训练数据集

```bash
# 0+1  状态/动作/子目标边界 + 跟踪（多张卡并行）；约 1 分钟一条示范
bash real_robot/run_track.sh  <yx_bowls 目录>  <RAW>  "0 1 2"
python real_robot/real_qa.py  <RAW>                      # 轨迹质检，必须全部通过

# 2-5  图像和标注 → 划分/归一化/配置 → 隐向量 → 检查
ASSEMBLE_ARGS="--steps 20000 --interval 5000 --devices 4 --per-device-batch 32" \
bash real_robot/run_build.sh  <RAW>  <DATASET>  code_causal_v3  <一份路径正确的方案 C 配置>
# 成功后 <DATASET>/CHECK_OK 存在，<DATASET>/contract_C_real.json 就是训练配置
```

两个脚本开头的路径变量（`PY`、`SIMPY`、`TRACKER_HOME`、`LIBERO_SIMULATOR_DRIVER_ROOT`、`PLAYWRIGHT_BROWSERS_PATH`、`TRACKER_SAM3_ROOT`）默认是我们集群上的位置，换机器请按注释设置环境变量。

## 训练

```bash
cd code_causal_v3
CUDA_VISIBLE_DEVICES=0,1,2,3 PI05_DEVICES=4 PI05_PER_DEVICE_BATCH=32 PI05_TOTAL_UPDATES=20000 PI05_CHECKPOINT_INTERVAL=5000 \
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONDONTWRITEBYTECODE=1 \
python -u train.py --contract <DATASET>/contract_C_real.json --dataset <DATASET> --output <OUT> --no-wandb
```

四个 `PI05_*` 必须和 `contract_C_real.json` 的 `training` 一致（`real_assemble.py` 结束时会打印应该用的值）。卡数、batch、步数要改，就用新的 `ASSEMBLE_ARGS` 重跑 `real_assemble.py`，再重跑 `real_check.py`。至少要 2 张卡（checkpoint 后的验证并行用 2 张）。

## 部署

```bash
# 自检 1：不需要模型和机器人。把一条录好的示范当成"实时相机"喂给部署感知链路，逐帧和训练图比对
python real_robot/real_deploy.py --code code_causal_v3 --replay <yx_bowls 目录> --episode 3 --compare <DATASET>

# 自检 2：加上模型，看它写出的记忆和动作误差（画面仍按录像走，不受模型控制）
python real_robot/real_deploy.py --code code_causal_v3 --replay <yx_bowls 目录> --episode 47 --checkpoint <CKPT>

# 真机：自己写一个类，提供 observe() 和 act(action14)，见 real_deploy.py 文件头
python real_robot/real_deploy.py --code code_causal_v3 --robot my_robot:Dobot --checkpoint <CKPT>
```

环境变量：`LIBERO_SIMULATOR_DRIVER_ROOT`（指向本仓库的 `renderer_driver/`）、`PLAYWRIGHT_BROWSERS_PATH`、`TRACKER_PYTHON` / `TRACKER_PYTHONPATH` / `TRACKER_SAM3_ROOT`（追踪器子进程）、`POLICY_PYTHON`（策略子进程，即训练环境的 python）。

## 异步部署（机械臂不停顿）：`real_deploy_async.py`

`real_deploy.py` 是同步的：执行完 20 个动作才问下一次，机械臂每 20 步会停下来等模型。想让机械臂边走边算，**不要**直接把询问间隔改短，要用 `real_deploy_async.py`。

**直接改短间隔为什么会坏。** 训练数据里每一帧 t 的"上一条记忆"都是 t-20 那一帧写的，Action 行描述的也是这 20 步。如果每 12 步问一次并把最新记忆喂回去，同一次抓/放会落进相邻两个窗口，模型给清单连打两个勾，提前写 `Task:Complete`。真机表现是"三次搬运只做了两次就停"。离线回放（验证集 5 条示范，图像按录像走，记忆用模型自己写的）：

| 询问间隔 | 喂给模型的记忆 | 清单超前的条数 | 结果 |
|---|---|---|---|
| 20 步 | 上一次（20 步前） | 0–1（下一次询问即对齐） | 5/5 在正确时间写完成 |
| 12 步 | 上一次（12 步前） | 1–2 | 4/5 清单超前，最早在 204 帧写完成（实际 277 帧结束） |
| 10 步 | **20 步前那次** | 0–1 | 5/5 在正确时间写完成 |
| 12 步 | **至少 20 步前的最近一次（24 步前）** | 0–1 | 5/5 在正确时间写完成 |

**`real_deploy_async.py` 做了三件事，让异步时模型看到的仍是训练时的情形：**

1. **记忆按 20 步取。** 每 `--period` 步（默认 10）发一次请求，但喂进去的记忆和隐向量来自"至少 20 步之前的最近一次请求"，不是最新那次。间隔为 10 时相当于两条交错的记忆链（0,20,40… 和 10,30,50…），每条都和训练完全一致。
2. **动作按请求时刻对齐。** 第 t 步发出的请求返回 t…t+19 步的动作；结果晚 d 步到达，就跳过前 d 个，在第 s 步执行第 s-t 个动作（动作是绝对关节目标，无需修正）。不会把过时的动作拿来执行。
3. **图像和状态取自同一步。** 在发请求那一步一起截取，再交给后台线程推理。

**时间要求：** 推理延迟（按控制步数计）≤ 20 − period。默认 period=10、30 Hz 时即一次推理 ≤ 0.33 秒；推理更慢就用更小的 `--period`（延迟 12 步 → `--period 8`；延迟超过 19 步则只能用同步）。脚本结束时打印实测延迟和 `starved steps`（没有新动作可用、保持上一个目标的步数），`starved` 不为 0 说明 period 设大了。

**没有解决的：** 相邻两段动作是各自独立采样的，交接处目标可能有小跳变。`--blend N`（默认 4）在新一段接管的前 N 步做线性过渡；彻底的做法是带前缀条件的采样（real-time chunking），需要改模型，这里没有做。

```bash
# 调度逻辑自测：不需要显卡、数据和机器人，几秒钟
python real_robot/real_deploy_async_selftest.py code_causal_v3

# 离线：画面按录像走，结果在请求后 8 步才"到达"
python real_robot/real_deploy_async.py --code code_causal_v3 --replay <yx_bowls 目录> --episode 47 --checkpoint <CKPT> --sim-latency 8

# 真机
python real_robot/real_deploy_async.py --code code_causal_v3 --robot my_robot:Dobot --checkpoint <CKPT> --period 10
```

验证范围：记忆规则在离线回放上验证过（上表后两行）；调度逻辑（动作对齐、记忆取用、停止时刻）用假模型自测通过。**完整脚本没有在真机上跑过，`--replay` 模式也没有带真模型端到端跑过**，第一次上真机请先低速、手放在急停上。日志在 `deploy_log/memory_async.jsonl`。

如果用的是自己写的异步循环，只需要移植第 1、2 条：维护"请求步 → (记忆, 隐向量)"的表，发请求时取至少 20 步前的最近一项；执行时按请求步对齐动作下标。

## 换一个任务或数据集要改什么

只改 `real_spec.py`：任务指令、裁剪框、物体名称和顺序、检测提示词、子目标条目及其对应物体。子目标边界目前只支持"抓-放"交替（由夹爪开合切分），别的任务结构需要改 `real_extract.py` 的 `segment()`。追踪器的三条修复规则是针对"两只一模一样的碗"写的，物体外观不同的任务通常用不到。
