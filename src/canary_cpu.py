"""只在笔记本 CPU 上验证代码机制有没有 bug。
不产出任何有效结果，禁止用它跑真实训练。"""

from src.train_cpt import run

if __name__ == "__main__":
    run("configs/canary_cpu.yaml")