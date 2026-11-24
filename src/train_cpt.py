import argparse
import yaml
from transformers import Trainer, TrainingArguments, default_data_collator

from src.data import load_packed_dataset
from src.model import load_model_and_tokenizer

def run(config_path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))

    model, tokenizer = load_model_and_tokenizer(cfg)
    dataset = load_packed_dataset(cfg, tokenizer)

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 20),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 250),
        save_total_limit=2,
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=cfg.get("seed", 42),
    )

    trainer = Trainer(model=model, args=targs, train_dataset=dataset,
                      data_collator=default_data_collator)
    trainer.train()
    trainer.save_model(cfg["output_dir"] + "/final")
    tokenizer.save_pretrained(cfg["output_dir"] + "/final")
    print("done ->", cfg["output_dir"] + "/final")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()