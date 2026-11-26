"""决策格式 SFT 入口：prompt/completion jsonl → completion-only 掩码 → 共用训练核心。"""

import argparse
import yaml

from src.model import load_model_and_tokenizer
from src.sft_data import load_sft_dataset
from src.train_cpt import train_core

def run_sft(config_path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    model, tokenizer = load_model_and_tokenizer(cfg)
    dataset = load_sft_dataset(cfg, tokenizer)
    train_core(model, tokenizer, dataset, cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_sft(args.config)

if __name__ == "__main__":
    main()