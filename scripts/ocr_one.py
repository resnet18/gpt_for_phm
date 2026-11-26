"""单文件 OCR：PDF（扫描版）→ data/processed/<文件名>.txt
用法：python scripts/ocr_one.py "files/某文件.pdf" [输出目录]
v2：坐标级双栏重排——rapidocr 默认按行横扫，双栏论文句子会被左右栏撕裂；
    这里按文本框的 x 中心分栏，栏内按 y 排序（先左栏后右栏），通栏块（标题/摘要/页眉）保持在前。
"""
import io
import sys
import pathlib

import fitz  # pymupdf
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

GUTTER_SCAN_STEP = 10  # 栏沟扫描步长（像素，dpi=200 下页宽约 1650px）

def reorder_columns(page_width, results):
    """results: rapidocr 输出 [框, 文本, 置信度]，box 为四角坐标。返回重排后的文本行列表。
    v3：找栏沟定栏——在页宽 15%~85% 区间扫描，被最少文本框跨越的 x 位置即栏沟；
        跨越栏沟的框=通栏（摘要/页眉），其余按中心分左右栏，各组内按 y 排序。"""
    items = []
    for item in results:
        box, text = item[0], item[1]  # 索引取值，兼容返回项长度
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "t": text})
    if not items:
        return []

    best = min(
        range(int(page_width * 0.15), int(page_width * 0.85), GUTTER_SCAN_STEP),
        key=lambda gx: sum(1 for it in items if it["x0"] < gx < it["x1"]),
    )
    full, left, right = [], [], []
    for it in sorted(items, key=lambda i: i["y0"]):
        if it["x0"] < best < it["x1"]:
            full.append(it)
        elif (it["x0"] + it["x1"]) / 2 < best:
            left.append(it)
        else:
            right.append(it)
    return [it["t"] for it in full + left + right]

def ocr_pdf(pdf_path, out_dir="data/processed", dpi=200):
    engine = RapidOCR()
    doc = fitz.open(pdf_path)
    out_path = pathlib.Path(out_dir) / (pathlib.Path(pdf_path).stem + ".txt")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    total_chars = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            result, _ = engine(img)
            lines = reorder_columns(img.width, result) if result else []
            total_chars += sum(len(l) for l in lines)
            f.write("\n".join(lines) + "\n")
    print(f"{pathlib.Path(pdf_path).name}: {len(doc)} 页, {total_chars} 字符 -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('用法: python scripts/ocr_one.py "files/某文件.pdf"')
    ocr_pdf(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/processed")