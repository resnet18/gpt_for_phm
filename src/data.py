from pathlib import Path

def load_packed_dataset(cfg, tokenizer):
    """读 data_dir 下所有 .txt，tokenize 后切成定长块。
    注意：整册读进内存，语料到几百 MB 规模没问题，上真语料要换 mmap/streaming。"""
    data_dir = Path(cfg["data_dir"])
    seq_len = cfg["seq_len"]

    texts = [f.read_text(encoding="utf-8") for f in sorted(data_dir.glob("*.txt"))]
    assert texts, f"no .txt found in {data_dir} — put some corpus there first"
    ids = tokenizer("\n".join(texts), return_tensors="np")["input_ids"][0]

    n = len(ids) // seq_len
    assert n > 0, f"corpus too small: {len(ids)} tokens < one block of {seq_len}"
    ids = ids[: n * seq_len].reshape(n, seq_len)
    return [{"input_ids": row.tolist(), "labels": row.tolist()} for row in ids]