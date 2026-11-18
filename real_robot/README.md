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

## 换一个任务或数据集要改什么

只改 `real_spec.py`：任务指令、裁剪框、物体名称和顺序、检测提示词、子目标条目及其对应物体。子目标边界目前只支持"抓-放"交替（由夹爪开合切分），别的任务结构需要改 `real_extract.py` 的 `segment()`。追踪器的三条修复规则是针对"两只一模一样的碗"写的，物体外观不同的任务通常用不到。
