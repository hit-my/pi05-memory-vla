# 真机外部智能体：维护记忆并从失败抓取中恢复

在 `real_deploy.py` 的部署循环外面加一个外部智能体。它不改策略的权重，也不看策略内部，只读策略写出的三行记忆和相机画面。

**状态：逻辑自测通过；真机、真实检查模型都还没有跑过。** 见文末"还没有验证的部分"。

## 它做三件事

1. **核查。** 策略每写一次 `Subgoal:done k`，就把"这一步开始时的俯视图、现在的俯视图、现在的腕部图"交给一个现成的视觉语言模型，问这一步是否真的完成。只核查物理步骤：`lift … cup`、`put … cup back`、`pick up block N`、`place the block on the plate`。
2. **维护记忆。** 判定"没完成"时，把第 k 项以及之后各项的勾从策略下一次读到的记忆里去掉，并把 `Task` 改回 `Incomplete`。策略看到这一步没完成，会重新去做。
3. **恢复。** 如果被撤销的是抓取类步骤（`lift` / `pick up`），并且夹爪读数表明手里没有东西，就先张开夹爪，再让机械臂沿它**自己记录下来的关节状态**原路退回到这次尝试开始的地方，然后把那时的记忆（去掉失败项的勾）交还给策略继续执行。

检查模型在后台线程里运行，不阻塞机械臂；判定结果在它返回后的下一次策略请求时生效。唯一的例外：策略写出 `Task:Complete` 后，机械臂原地等待尚未返回的判定（最多 `--verdict-timeout` 秒），判定为"没完成"时取消停止。

检查模型出错（断网、返回格式不对）只记日志，不会中断运行，也不会改记忆。

## 文件

| 文件 | 内容 |
|---|---|
| `real_agent.py` | 核查调度、记忆修改、恢复路径。没有机器人、相机和网络代码 |
| `real_checker.py` | 检查模型的提问方式（三图拼接 + 提示词）和调用 |
| `real_deploy_agent.py` | 部署循环：`real_deploy.py` 的真机模式加上智能体 |
| `real_agent_selftest.py` | 用替身机器人、替身策略、替身检查模型做的逻辑自测 |

## 用法

```bash
# 检查模型：任意兼容 OpenAI chat-completions 接口的视觉语言模型。密钥只放环境变量，不要写进文件或命令行
export CHECKER_API_URL=...   CHECKER_MODEL=...   CHECKER_API_KEY=...

python real_robot_blocks/real_agent_selftest.py                         # 先跑自测，不需要数据和网络

python real_robot_blocks/real_deploy_agent.py --robot my_robot:Dobot --checkpoint CKPT --code CODE --no-agent     # 基线，等同 real_deploy.py
python real_robot_blocks/real_deploy_agent.py --robot my_robot:Dobot --checkpoint CKPT --code CODE --no-recover --save-dir checks   # 只维护记忆，智能体不动机械臂
python real_robot_blocks/real_deploy_agent.py --robot my_robot:Dobot --checkpoint CKPT --code CODE --save-dir checks                # 加上恢复
```

机器人类与 `real_deploy.py` 相同：`observe() -> {"top", "right_wrist", "state"[14]}`，`act(action14)`。其余环境变量（追踪器、渲染器、策略的 Python）也与 `real_deploy.py` 相同。

日志：`deploy_log/memory.jsonl`（每次策略请求）、`deploy_log/agent.jsonl`（每次核查、撤销、恢复）。`--save-dir` 会把每次提问的拼图、提示词和回答存下来，可用 `real_checker.py --case` 重放，用来离线调提示词和阈值。

主要参数：

- `--threshold`（默认 0.5）：P(完成) 低于它才撤销。
- `--wait-calls`（默认 1）：策略打勾后隔几次请求再看结果。策略在夹爪闭合时就打勾，物体是否被提起要稍后才看得出来。
- `--back-calls`（默认 5）：恢复时退回到错误打勾之前第几次请求的位置，但不早于这一步开始的位置。
- `--rewind-speed`（默认 1.0）：退回时每步关节变化量的上限，以"来的路上最快那一步"为 1。
- `--fix-observations`：另外核查"`X cup: has block / empty`"这类观察项，检查模型明确反对时把它改写。**默认关闭**，见下。

## 安全

恢复期间机械臂不受策略控制。它只回到自己到过的状态、夹爪张开、不比来时更快，但**不会看到这段时间里进入工作区的新东西**。请先用 `--no-recover` 跑，确认核查结果可信，再打开恢复，并始终把急停放在手边。

以下情况智能体不会动机械臂，只改记忆：

- 被撤销的不是抓取类步骤（放回、放置）。
- 夹爪读数在 0.45–0.80 之间，即手里可能夹着东西（杯子约 0.67，数字方块约 0.55）。这时张开夹爪可能把东西从高处扔下。
- 没有可以退回的更早记录。

## 关于 `--fix-observations`

这个任务的答案取决于"看到几个粉色方块"，所以观察项写错是最致命的错误。打开这个选项后，智能体会在策略写下观察项时问检查模型"杯子原来的位置上有没有粉色小方块"，只有当检查模型给出明确相反的结论（偏离 0.5 至少 0.3）时才改写。

它默认关闭，原因是：这等于让外部模型替策略做了任务里最关键的判断，和"只核查已完成的步骤"不是同一类干预。做实验时应该把它作为单独的一组来报告，不要和默认配置混在一起。

## 还没有验证的部分

- **真机没有跑过。** `real_agent_selftest.py` 只验证调度逻辑：错误的勾会被撤销、恢复时夹爪张开且不超速、机械臂回到记录的位置、策略重做后任务完成、停止会等待最后的判定、检查模型失败不影响运行。
- **检查模型在这个场景下的准确率未知。** 提示词是照仿真实验里效果最好的那版改写的，没有在真实画面上测过。建议先用 `--no-recover --save-dir` 录几条，人工看一遍 `checks/` 里的判定，再定阈值。
- **夹爪读数的阈值（0.45–0.80）来自示范数据的统计**，"闭合但什么都没夹到"时的读数没有实测过。
- **恢复后追踪器的状态没有重置。** 抓取失败时物体没有被移动，所以静止位置仍然正确；如果失败的抓取把物体碰歪了，轨迹会有一次小的跳变。
- **只支持同步循环**（策略计算时机械臂等待）。还没有和 `real_deploy_async.py` 的无停顿循环合并。
- 没有做仿真实验里的"定期复查最近一项"。
