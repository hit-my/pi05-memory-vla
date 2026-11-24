# agent_eval：仿真评测时的外部清单检查（失败恢复 agent）

只在**评测时**套在模型外面，训练不需要任何改动：同一个存档可以带它测，也可以不带它测。

## 它做什么

模型每次给清单打勾（`Subgoal:done k`）时，一个与训练无关的视觉语言模型看三张图（该步开始时的主视角、现在的主视角、现在的腕部视角），判断第 k 步是否真的完成。判为没完成时：

1. 把这一步（以及之后的步）的勾从喂回给模型的记忆里去掉，`Task` 置为 `Incomplete`，模型于是把这一步当作没做、重新去做；
2. 可选的物理恢复（`STUDY_REC=back`）：先张开夹爪，再沿机械臂自己走过的路径退回到错误打勾之前的位姿（只用策略自己记录的末端位姿，不读仿真器状态），并把当时的记忆还给模型。

检查不阻塞策略：仿真不是实时的，所以延迟按"策略调用次数"建模，第 n 次调用发起的检查在第 n + `STUDY_AGENT_DELAY` 次调用时才生效（一次调用 = 20 个控制步）。唯一的例外是模型宣布整个任务完成时，会等最后一个勾的检查结果再停。

它**不使用仿真器真值**，也不改模型权重，只改模型读到的记忆文本（以及可选的那一小段回退动作）。

## 结果（方案 C，5 万步存档，26 个任务 × 10 局，种子 50–59）

| 设置 | 成功局数 | 成功率 |
|---|---|---|
| 标准评测（不带 agent） | 140/260 | 53.8% |
| 带 agent，回退 2 次调用（约 40 帧） | 156/260 | 60.0% |
| 带 agent，回退 4 次调用（约 80 帧） | 159/260 | 61.2% |

40 帧和 80 帧之间的差别在噪声范围内。带 agent 的两次运行所用的环境变量：

```
STUDY_AGENT_TH=0.5 STUDY_AGENT_PERIOD=0 STUDY_AGENT_DELAY=5 STUDY_AGENT_WAIT=0
STUDY_AGENT_VERBS=pick,place,open STUDY_AGENT_TWOLOOK=pick,place,open STUDY_AGENT_LAG2=3
STUDY_REC=back STUDY_REC_ANCHOR=close STUDY_REC_GAIN=3.2 STUDY_REC_BACK=2    # 80 帧那次是 STUDY_REC_BACK=4
```

## 文件

| 文件 | 作用 |
|---|---|
| `policy_agent.py` | 策略外层：发起检查、按结果改记忆、可选的物理恢复。全部开关见文件头的环境变量说明 |
| `checker_omp.py` | 检查服务：`POST /<TOKEN>/judge`，返回 `{"p_yes", "seconds"}`。这一版通过本机的 `omp` 客户端调用视觉语言模型，并可反向隧道到评测服务器 |
| `glm_full.py` | 检查用的提示词（`checker_omp.py` 只导入其中的提示词构造函数）；单独运行时是在 230 个带真值的打勾事件上离线测试提示词 |
| `run_agent_eval.sh` | 启动带 agent 的标准评测 |
| `paired.py` | 两次评测的同局配对比较（成功率、新赢/新输、分任务） |

## 怎么跑

1. 复制一份 `code_causal_v3`，把 `policy_agent.py` 放进去，并在 `policy_ens.py` 的 import 区加一行 `import policy_agent`。不设 `STUDY_AGENT=1` 时这份代码的行为和原来完全一样。
2. 启动检查服务（任何实现了同样 HTTP 接口的服务都可以）：`python checker_omp.py <工作目录> <模型名> <TOKEN>`。
3. `CODE=<上面那份拷贝> EVAL=<code_eval_v5> CKPT=<存档> OUT=<结果目录> bash run_agent_eval.sh fullClose40 1,2,...,26 STUDY_AGENT_URL=http://127.0.0.1:18780/<TOKEN> <上面的环境变量>`
4. `python paired.py <不带 agent 的结果目录> <OUT>/arm_fullClose40`

## 没有做的

- **真机上没有这套检查。** `real_robot/` 和 `real_robot_blocks/` 的部署脚本里，清单完全由模型自己打勾。把它接到真机需要：真机能访问检查模型、处理检查耗时、以及一个不依赖仿真的回退动作。
- 只在一个存档（5 万步）上测过；检查模型换了以后阈值需要重新标定。
- 训练示范里没有"失败后重来"的例子，模型靠自己从被退回的状态重新执行。