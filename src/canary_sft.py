"""只在笔记本 CPU 上验证 SFT 管线（尤其掩码逻辑）有没有 bug。
不产出任何有效结果，禁止用它跑真实训练。"""

from src.train_sft import run_sft

if __name__ == "__main__":
    run_sft("configs/canary_sft.yaml")