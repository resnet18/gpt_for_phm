import json
from pathlib import Path

def load_sft_dataset(cfg, tokenizer):
    """读 data_dir 下所有 .jsonl（每行 {"prompt":..., "completion":...}），
    prompt 与 completion 拼接为一条序列，prompt 部分 labels=-100（loss 只算 completion）。
    completion 末尾追加 EOS——不教终止符的话，推理时模型报完数不知道停，会复读尾词。
    输出统一填充到 seq_len：input_ids 用 pad_token、labels 用 -100、附 attention_mask。
    超长样本直接丢弃（canary/构造数据阶段可接受，真数据阶段再定截断策略）。"""
    data_dir = Path(cfg["data_dir"])
    seq_len = cfg["seq_len"]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id  # GPT-2 系无 pad_token，退化为 eos
    eos_id = tokenizer.eos_token_id

    rows = []
    for f in sorted(data_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert rows, f"no .jsonl found in {data_dir} — put some prompt/completion pairs there first"

    samples = []
    for r in rows:
        p_ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(r["completion"], add_special_tokens=False)["input_ids"] + [eos_id]
        n = len(p_ids) + len(c_ids)
        if n > seq_len:
            continue
        # 防静默失效断言：prompt 段必须全被掩码，completion 段（含 EOS）必须一个都不被掩。
        # 掩码写反/错位时 loss 照样下降，这两条断言是唯一会当场叫的东西。
        labels = [-100] * len(p_ids) + c_ids
        assert all(x == -100 for x in labels[:len(p_ids)]), "prompt 未完整掩码"
        assert not any(x == -100 for x in labels[len(p_ids):]), "completion 被误掩"
        samples.append({
            "input_ids": p_ids + c_ids + [pad_id] * (seq_len - n),
            "labels": labels + [-100] * (seq_len - n),
            "attention_mask": [1] * n + [0] * (seq_len - n),
        })
    assert samples, "all samples dropped (longer than seq_len)"
    return samples