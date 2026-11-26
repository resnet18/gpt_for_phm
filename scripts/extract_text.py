"""文字层提取：文字版/图文混排 PDF 直接提文本，提不出字数的页标记为待 OCR 回填。
用法：python scripts/extract_text.py "data/incoming/某文件.pdf" [输出目录]"""
import sys
import pathlib

import fitz  # pymupdf

MIN_CHARS = 50  # 一页提取字符低于此数 → 视为无文字层页，交给 OCR 回填

def extract(pdf_path, out_dir="data/processed"):
    doc = fitz.open(pdf_path)
    out_path = pathlib.Path(out_dir) / (pathlib.Path(pdf_path).stem + ".txt")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    total, low_pages = 0, []
    with open(out_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(doc):
            t = page.get_text().strip()
            total += len(t)
            if len(t) < MIN_CHARS:
                low_pages.append(i + 1)
            f.write(t + "\n\n")
    flag = low_pages if low_pages else "无"
    print(f"{pathlib.Path(pdf_path).name}: {len(doc)} 页, {total} 字符, 缺字页: {flag} -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('用法: python scripts/extract_text.py "data/incoming/某文件.pdf"')
    extract(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/processed")