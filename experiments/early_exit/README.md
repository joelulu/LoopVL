# LoopVL H1/H2 最小验证（免训练）

此目录使用原始 H2L3 权重和原始完整深度 greedy 输出作参考。
不训练新模型，不改变 target 的输出分布，不是 LoopSpec 流水线实现。
所有实验 forward 都关闭 KV cache；本轮只验证候选质量及 greedy 正确性，
**diagnostic_seconds 不能解释为实际投机加速比**。

## 安装

建议使用独立 Python 3.12 环境，与 LoopSpec 的 Python 3.11 环境分开。
先安装适合服务器 CUDA 驱动的 torch/torchvision，再安装仓库原有依赖：

```bash
python -m pip install -r requirements.txt
hf download TierFlow/LoopVL --local-dir model
python scripts/verify_repo.py --verify-model
```

权重目录也可以使用已有的完整模型快照。不要对整个模型调用 `.half()` /
`.bfloat16()`：原始 loader 保留 BF16 核心与 FP32 adapter 的混合精度。

## 输入

在仓库根目录准备 `samples.jsonl`，每行是一条独立图文请求：

```json
{"id":"chart01","image":"images/chart01.png","prompt":"What is the highest value in the chart?","category":"chart"}
{"id":"math01","image":"images/math01.png","prompt":"Solve the problem in the image. Explain briefly, then give the answer.","category":"math"}
```

这是格式示例，需要填写真实图像路径；相对路径默认相对 JSONL 所在目录，
也可传 `--image-root /absolute/dataset/root`。保留各任务原提示，不自动加 CoT。
首轮选 8 条 smoke；随后建议 96 条（普通理解、OCR/图表、数学/空间各32条）。
程序不会隐式下载、补齐或重复样本，summary 报告实际样本和 token 数。

## A：完整轨迹下的浅层读出

```bash
python -m experiments.early_exit.probe \
  --model-dir ./model --input samples.jsonl \
  --output runs/h1h2_smoke --limit 8 --max-new-tokens 128
```

1. 原始 `greedy_generate` 生成参考回答。
2. 以原 PrefixLM mask 重放参考回答（最后一个标签不作为输入）。
3. hook **整个 H_module 的输出**，取得 H1/H2，不重复做 final_norm。
4. 正确移动一位，比较每个下一 token；分块投影 logits，避免保存整段巨大词表。

`tokens.csv`：H1/H2 token、原生成 token、H1概率及top1-top2概率差。
`summary.json`：逐任务和全局 token 加权一致率、高置信不一致率、H2重放错误数。
`generations.jsonl`：实际输入、生成文本/token、视觉 token 数。
`settings.json`：模型路径、git revision、运行环境及参数。

默认 H2 重放与原生成不一致会停止；这可能暴露 mask/位置错误或 BF16
不同计算 shape 的数值差异。先检查 token 和 margin，不要直接宣称 lossless。
`--allow-replay-mismatch` 仅允许继续收集诊断，不会抹去错误。
这些都是同一正确前缀下的读出，连续一致区间不代表真实 drafting 接受长度。

## B：H1 实际连续猜测，再完整验证

```bash
python -m experiments.early_exit.draft_verify \
  --model-dir ./model --input samples.jsonl \
  --output runs/h1_draft_smoke --limit 8 --max-new-tokens 128 \
  --draft-lengths 1 2 4
```

B 会先执行 A，再从每个已提交前缀出发，让 H1 使用自己的候选继续生成。
H1-only 通过临时 forward hook 在第一次完整 H 调用后停止，不执行 H2，
也不修改 H_cycles 或视觉重注入配置。hook 总会在 finally 中移除。
验证一次处理整段候选，返回 K+1 个因果对齐的预测；只提交匹配前缀与
一个 correction/bonus，并正确处理 EOS 和剩余输出预算。

`draft_verify.jsonl` 保留每轮候选、target预测、接受边界及最终输出。
`draft_summary.json` 是每个样本/长度的简表，`tokens_per_verification`
包含 correction/bonus；`accepted_draft_tokens` 只计真正提交的候选。
最终 token 必须逐个等于原始 greedy 输出，否则保存错误记录后停止。
没有 `COMPLETE.json` 说明运行未完整通过。

计时不做 warmup、不做速度比较；图像编码复用范围也与原生成不同。
后续先做正确的缓存 AR，再比较同缓存后端的实际流水线性能。

## 本地协议测试

```bash
python -m unittest experiments.early_exit.test_core -v
python -m unittest experiments.early_exit.test_runtime_contract -v
```

这些测试不需要 torch；runtime contract 测试需要 numpy。覆盖依赖历史的
toy target、错误传播、首拒绝、全接受 bonus、EOS、输出预算，以及H1提前停止、
异常后的hook清理和PrefixLM mask。实际模型测试需服务器权重/GPU。

## Expanded H1/H2 readout diagnostic

```bash
.venv/bin/python -m experiments.early_exit.make_readout_samples --output runs/readout_inputs_new
.venv/bin/python -m experiments.early_exit.probe \
  --model-dir "$HOME/LMM/lmm_model/LoopVL" \
  --input runs/readout_inputs_new/samples.jsonl \
  --output runs/readout_probe_new --device cuda:0 --limit 20 --max-new-tokens 32
```

Creates **10 unique synthetic images, 20 questions**: five charts and five counting
images, each with a short-answer and descriptive question. No natural-image data
or benchmark accuracy claim. Longer prompts need not yield long answers; actual
generated lengths are saved. Output/input directories must be new.

The checkpoint's H2L3 schedule is unchanged: LLLHLLLH, 16 layers per stack call;
H1 follows application 64 and H2 follows application 128 (excluding vision).
This is two high cycles, not two Transformer layers. Further H cycles would
change the target model and are outside this diagnostic.

The probe now records decoded tokens including EOS, H2 confidence, the H2 token's
rank/probability in H1, Top-5/10 membership, hidden RMS/cosine, and top candidates.
`readout_summary.json` separates all/content/EOS/first positions; `summary.json`
retains the previous aggregates. H2 is not a ground-truth answer; these are
readout agreements, not answer accuracy or speculative acceptance rates.

By default up to four evenly spaced positions per sample additionally run H1-only
on the exact available reference prefix (never the predicted token or its suffix).
`h1_prefix_checks.jsonl` compares that top-1 with the H1 captured during causal
full-sequence replay. A mismatch stops the run; inspect masking and BF16 shape
sensitivity before interpreting it. `--check-h1-positions 0` disables checks and
`--allow-replay-mismatch` permits diagnostic continuation while retaining records.
No threshold is lowered to manufacture confidence. No speedup is reported.

CPU tests: `python -m unittest experiments.early_exit.test_readout_metrics -v`.

Readout schema v2 fixes a diagnostic bug: greedy IDs now use logits.argmax,
matching actual decoding. torch.topk may choose a different ID among tied maxima;
its ordering must not define Top-1 agreement. This could falsely lower Ouro
cycle4 self-agreement and falsely fail LoopVL H1 prefix checks. Top-5/10 still
use actual topk candidate sets; tied boundary membership can be arbitrary.
Rerun diagnostics into a new output directory; weights and inputs are unchanged.
