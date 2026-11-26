## 2026-11-25
- canary @笔记本CPU | config: canary_cpu.yaml (sshleifer/tiny-gpt2, seed 27355) | 5 步 / 1.1s | loss 10.8 ≈ ln(50257)=均匀预测基线，5 步 warmup 内预期内 | grad_norm 非零且波动，梯度流通 | 产物 outputs/canary_cpu/final | 结论：代码机制 e2e 通过；loss 是否真降属 5060Ti 冒烟指标，本测试不作要求
- smoke_0.6b @工位机 5060Ti（16G）| Qwen3-0.6B-Base 全参 CPT，seed 27355 | 500 步 / 27.7min | loss 2.90→0.0014 单调下降 | 吞吐≈4.9k tok/s | 结论：CPT 管线验收通过
  - 解读：有效信号在前 50 步（2.9→0.5）；终值趋零=45 epoch 单书（Gutenberg 小说）记忆，预期内，冒烟不考核终值
  - 环境教训：torch 从 PyPI 裸装会被解析成 CPU 版，必须显式 cu132 索引（本机 2.15.0+cu132 / CUDA 13.2 / 驱动 616.64）；transformers 新版 gradient_checkpointing 参数名 deprecation 警告无害

## 2026-11-26
- canary_sft @笔记本 CPU | sshleifer/tiny-gpt2，seed 27355 | 5 步 / 2.4s | loss 10.83 = ln(50257) 均匀基线（warmup 内预期读数）| grad_norm 非零 | 掩码断言×2 + 变长填充逻辑全部通过 | 产物 outputs/canary_sft/final | 结论：SFT 管线机制验收通过
  - canary 抓获记录①：transformers 新版 Trainer 强制 accelerate&gt;=1.1.0（环境层缺失，pip 补齐）
  - canary 抓获记录②：SFT 变长样本直接进 default_data_collator 触发 ragged tensor 报错 → sft_data.py 增加三填充（input_ids=pad_token、labels=-100、attention_mask），修复后通过

## 2026-11-26
- smoke_sft @工位机 5060Ti | Qwen3-0.6B-Base + toy 8 条决策格式样本，seed 27355 | 20 步 / 22s | loss 2.31→0.051 | 结论：SFT 管线验收通过
  - 格式抽查：待判样本续写输出"决策：香蕉。置信度：0.95。&lt;|endoftext|&gt;"——格式成立、EOS 终止生效；决策内容错（8 条记忆检索）属预期，不考核
  - 关键修复：completion 末尾必须拼 eos_token_id，否则推理复读尾词（sft_data.py，已入 README 格式宪法）；变长样本三填充（input_ids=pad / labels=-100 / attention_mask）
  - 环境备注：transformers 新版将 gradient_checkpointing 改名为 activation_checkpointing，train_cpt.py 用 try/except 自适应新旧版本