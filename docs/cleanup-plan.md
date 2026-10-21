# 代码仓库整理计划

目标：读者从 GitHub 获取代码、从 Hugging Face 获取 checkpoint 后，能完成数据读取、生成、特征计算和成员推断评估。公开仓库不需要附带论文结果、完整生成缓存或训练日志。

## 本次核对范围

- GitHub 基线：`main`，提交 `7f056254fae0015e472ac6f956229b376bb2c2c4`。
- 服务器源码：`/mydata/rlmia/Divergence-in-Behavior-Attack`。
- 服务器入口：`/mydata/rlmia/run_dib_attack.sh`。
- 训练和数据构建：`/mydata/rlmia/grpo_run`，含 `toolrl_run` 子目录。
- 模型发布：RLMIA 组织下的 18 个独立仓库。已核对 repo ID、revision、根目录配置和文件清单，见 [configs/models.json](../configs/models.json)。

本次在独立本地分支整理；服务器原仓库及训练目录保持原状。运行检查使用独立临时目录，读取已有 checkpoint 和数据。

## 本次已落实

- [x] README：安装、数据字段、模型选择、最小 GPU 检查、完整运行命令、CPU 缓存评估及输出说明。
- [x] 分离 CPU 依赖和 GPU 运行依赖，记录实测环境。
- [x] 新增 `audit.py`：用 Transformers 完成文本 MATH 流程，不依赖服务器专用路径或 vLLM。
- [x] 按模型分进程生成，避免上一阶段的 GPU 资源残留。
- [x] 按 `row_id/sample_id` 配对响应，检查缺失和重复；按数据哈希与运行配置检查缓存。
- [x] 修复 MATH 提示要求 `####`、旧评分器仅识别 `\boxed{}` 的格式不一致。
- [x] 将 token 统计转为 FP32，并限制极端 log-ratio；对应服务器新版特征脚本的数值稳定处理。
- [x] CPU 冒烟脚本及 GPU 最小流程检查，具体证据见 [validation.md](validation.md)。
- [x] 忽略 runs、results、checkpoints、缓存和日志，保留已有数据文件。
- [x] 核对 18 个公开模型仓库，README 使用实际 repo ID 和固定 revision；默认从 Hugging Face 官方下载。

历史 `main.py` 及研究脚本暂时保留。新版文本入口只验证了 MATH；没有把未检查的扩展任务标成可复现。

## 文件去向

以下为后续目标布局，不代表这些目录已经存在：

```text
README.md
requirements-cpu.txt
requirements.txt
audit.py
src/
  data.py                  # 统一 Parquet/响应 schema
  generation/              # Transformers；可选 vLLM/SGLang
  features/                # token 特征与轨迹 mask
  verifiers/               # math、code、instruction、tool、search
  evaluation/              # unified、baseline、transfer
scripts/
  smoke_test.py
  prepare_data/
  training/
configs/
  models.json              # repo/revision/config 已落实；reference/split 待补
  tasks/
docs/
  cleanup-plan.md
  validation.md
dataset/
runs/                      # 忽略
checkpoints/               # 忽略
```

先验证功能，再迁移目录，并保留旧入口的兼容跳转，避免一次性移动所有文件导致相对导入和命令同时失效。

| 当前文件/位置 | 建议归属 | 要做的工作 |
| --- | --- | --- |
| `main.py`、`src/feature_computation.py` | 统一 audit 入口和 generation/features 模块 | 合并新旧入口；评分模式显式参数化；vLLM 作为可选 backend |
| `src/compute_score.py`、`ifmulti.py`、`code_score.py` | `src/verifiers/` | 明确任务输入 schema；隔离各自依赖；保留第三方版权头 |
| `src/evaluation.py` | `src/evaluation/` | 移除未运行 specialist 的空报告；为实验配置提供明确输出 |
| `indivdual.py`、`baselines.py` | evaluation 子命令 | 修正命名时保留兼容入口；补 CLI 检查和依赖 |
| `plot.py`、`convert.py` | 可选分析/转换脚本 | 从复现主路径移出；检查硬编码路径，不携带结果文件 |
| `dataset/math-shadow/*1.parquet` | 数据清单标记重复，之后去重 | 已核对两组文件 SHA-256 相同，不能作为独立 split |
| 服务器 `src/toolrl_score.py` | `src/verifiers/toolrl.py` | 将 ToolRL ground truth 格式和计算逻辑一起迁移 |
| 服务器 `search_r1_mia.py` | Search-R1 扩展入口 | 拆分生成、检索、feature mask、评分和评估 |
| 服务器 `eval_*transfer*.py` | transfer 子命令 | 明确 source/target/split 配对，去掉绝对路径 |
| 服务器 `eval_math3b_multiseed.py`、`eval_cached_feature_ablation.py` | 可选实验驱动 | 参数化缓存位置和 seed，避免复制主算法 |
| `grpo_run/build_*.py`、`prepare_*.py` | `scripts/prepare_data/` | 固定数据源 revision、seed 和 split manifest |
| `grpo_run/run_*.sh`、`toolrl_run/*` | `scripts/training/` | 训练与审计环境分开；参数化模型、设备、数据和输出路径 |

## P0：补齐模型与数据对应关系

这是公开“一条命令运行”示例前最需要补齐的部分。

1. 已核对 RLMIA 组织下的 18 个独立模型仓库及其根目录配置，记录在 `configs/models.json`。
2. 继续为每个仓库补齐经过确认的基座 reference、训练数据 split、RL 配置和可用审计入口；现有配置与文件清单不能单独证明训练来源。
3. README 已用 `RLMIA/GRPO-3B-MATH` 及固定 revision 替换占位目录。模型权重和 tokenizer 的完整下载/加载测试仍需按发布仓库进行，不能与服务器本地 checkpoint 的 smoke 记录混为一谈。
4. 为数据附上来源、版本、行数、SHA-256、划分 seed 和 member/nonmember 定义。HF 模型与训练 split 必须能一一对应。
5. 单独记录运行环境和训练环境。检查中的服务器 `run_slime_math_3b.sh` 使用 8 个训练 rollout、4,096 最大响应长度；这不能自动当作论文已确认的 16/8,192 主实验配置，也不能直接改写成后者。需要按 checkpoint 来源区分。

验收：选择一个发布模型后，能下载正确的模型和 reference、选中正确 split，并完成 README 的小规模 GPU 检查。

## P1：稳定主入口和依赖

1. 将 `audit.py` 中已验证的参数、manifest 和响应配对逻辑合并到统一入口；保留旧参数别名。
2. 评分器用显式 `--task` 分派，去掉手改源码的模式切换。公开版 `feature_computation.py` 固定为 math，服务器工作区固定为 toolrl。
3. vLLM、SGLang、VLM 和代码执行依赖分别声明；CPU 特征缓存评估不应导入 GPU 框架。
4. 增加受测的批量生成 backend；保持采样配置可追溯，不默认宣称不同 backend 逐 token 一致。
5. 缓存记录模型 revision、tokenizer/chat-template 信息、feature schema 版本和 verifier 版本；当前实现只记录模型位置，不对可变目录中的权重做校验。
6. 明确失败策略：空响应、超长上下文、NaN/Inf、截断响应不能静默转成“成功复现”；保留可定位的错误和重新运行说明。

验收：干净 Python 3.11 环境安装成功；CPU smoke 无 GPU 依赖；每个支持的 backend 跑通同一小样本并生成一致格式的输出。

## P2：迁移服务器上的扩展

### ToolRL

来源：`run_dib_attack.sh`、`src/toolrl_score.py`、`dataset/toolrl/`、`grpo_run/toolrl_run/`。

- 保留完整 system/tool 描述及多轮对话，不能沿用旧版只取第一条 content 的处理。
- 提供 member/nonmember 数据构建脚本和明确的来源对应关系。
- 模型必须是已合并 checkpoint；训练的 LoRA 保存目录和审计目录需要区分。
- 替换训练脚本按设备批量 kill 进程的逻辑，只管理脚本自己启动的进程。
- 验证一个小型工具调用样本的生成、解析和打分。

### Search-R1

来源：`search_r1_mia.py`、`grpo_run/prepare_search_r1_mia_data.py`、`build_hotpot_member_nq_nonmember.py`、`build_search_r1_mixed_nonmember.py`、`retrieval_server_mmap.py`。

- 目前通过 SGLang `/generate` 和检索服务 `/retrieve` 运行，需要独立说明服务启动、端口、索引与语料获取。
- 去掉 `/root/slime/examples/search-r1` 的隐式导入，将评分器作为显式依赖或在许可允许时单独收录。
- 保留“只计算模型生成 token”的 mask；检索得到的 information 文本不应混入被审计响应的 token 统计。
- 统一重试、截断、abort、非法动作的处理，并报告保留的样本数。
- `--score-mode math` 只改变特征阶段的评分器，不会把 Search-R1 的生成协议变成普通 MATH 生成。不能据此把 `--stage all` 当作 MATH 一键入口。
- 多 GPU feature shard 合并后必须检查重复和缺失键。

### MATH behavior / shadow / model-family transfer

来源：`eval_math_behavior_mia.py`、`eval_math3b_multiseed.py`、`eval_fixed_target_shadow_transfer.py`、`eval_shadow_transfer_matrix.py`、`eval_model_family_transfer.py`。

- `eval_math_behavior_mia.py` 依赖外部 `grpo_run.math_reward` 和 `slime.rollout.rm_hub.math_utils`。先明确训练 reward 与发布 verifier 的差异，再决定迁移方式。
- 区分固定 checkpoint 的 predictor cross-validation、重新训练 RL 模型、多 seed predictor、shadow transfer；配置中不要混用这些概念。
- matched split B 由 `build_math_matched_split_b.py` 构建，保留数据源 revision、排除集合及 manifest。
- 只迁移脚本和必要数据说明，不复制历史生成缓存和报表。

### Geo3K / code / instruction following

- Geo3K 已有图像 Parquet，但本次文本入口主动拒绝图像数据。需要单独检查 Qwen-VL processor、image bytes 和多模态特征前向。
- code/IF 当前缺少完整的公开数据准备链。先补来源与 schema，再提供运行命令。
- VLM/code/IF 各做一个独立 smoke fixture，验证后再加入 README 支持列表。

## P3：发布验收

- [x] 验证 18 个 HF repo ID、revision、配置和文件清单。
- [ ] 为发布模型确认 reference 与数据 split 映射，并完成权重下载/加载检查。
- [ ] 在干净环境照 README 安装并运行一次。
- [ ] 已声明支持的任务全部有可执行的小规模示例；无外部私有路径。
- [ ] 缓存输入 schema、特征名称顺序和版本固定。
- [ ] 生成式和缓存式两条路径均有 smoke 检查；无需发布实验结果。
- [ ] Git 中无新增权重、生成缓存、日志或凭据；第三方许可和版权信息保留。

暂不需要提交：`results/`、`grpo_run/out/`、训练日志、ROC 图、旧 ZIP、完整生成 JSONL、模型权重。它们可以继续保留在研究服务器或相应模型托管平台。
