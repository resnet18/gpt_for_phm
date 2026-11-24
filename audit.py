#!/usr/bin/env python3
"""
轻量数据源审计 - 只扫单层 files/ 目录
pip install pymupdf
"""

import os
import csv
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit("先装 PyMuPDF: pip install pymupdf")

ROOT = Path("files")  # 改成你的实际路径
OUTPUT = "audit.csv"


def check_pdf(pdf_path: Path) -> tuple:
    """返回 (页数, 预估类型, 备注)"""
    try:
        doc = fitz.open(pdf_path)
        pages = len(doc)
        
        # 只检测前3页文本量
        text = ""
        for i in range(min(3, pages)):
            text += doc[i].get_text()
        doc.close()
        
        char_per_page = len(text) / min(3, pages) if pages > 0 else 0
        
        if char_per_page > 300:
            return pages, "文字版PDF", f"前3页平均每页{int(char_per_page)}字符"
        elif char_per_page > 30:
            return pages, "图文混排PDF", "有文字层但较少，可能含大量图片/公式"
        else:
            return pages, "扫描版PDF", "几乎无文字层，需OCR或多模态"
            
    except Exception as e:
        return 0, "PDF损坏", str(e)[:30]


def audit():
    rows = []
    summary = {"文字版PDF": 0, "扫描版PDF": 0, "图文混排PDF": 0, 
               "Word": 0, "PPT": 0, "Excel": 0, "图片": 0, 
               "压缩包": 0, "其他": 0, "损坏": 0}
    
    files = sorted(ROOT.iterdir()) if ROOT.is_dir() else []
    
    print(f"扫描目录: {ROOT.absolute()}")
    print(f"共 {len(files)} 个文件\n")
    print("-" * 70)
    
    for f in files:
        if not f.is_file():
            continue
            
        ext = f.suffix.lower()
        size_mb = f.stat().st_size / (1024 * 1024)
        
        # 默认分类
        ftype, note, pages = "其他", "", 0
        
        if ext == ".pdf":
            pages, ftype, note = check_pdf(f)
            if "损坏" in ftype:
                summary["损坏"] += 1
            else:
                summary[ftype] += 1
                
        elif ext in (".doc", ".docx"):
            ftype, note = "Word", "python-docx/docx2txt提取"
            summary["Word"] += 1
            
        elif ext in (".ppt", ".pptx"):
            ftype, note = "PPT", "python-pptx提取或转图片"
            summary["PPT"] += 1
            
        elif ext in (".xls", ".xlsx", ".csv"):
            ftype, note = "Excel", "pandas读取"
            summary["Excel"] += 1
            
        elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"):
            ftype, note = "图片", "需OCR或多模态模型"
            summary["图片"] += 1
            
        elif ext in (".zip", ".rar", ".7z", ".tar", ".gz"):
            ftype, note = "压缩包", "需解压后处理"
            summary["压缩包"] += 1
            
        elif ext in (".txt", ".md", ".json", ".xml", ".html", ".htm"):
            ftype, note = "纯文本", "直接读取"
            summary["文字版PDF"] += 1  # 复用统计，表示可直接处理
            
        else:
            summary["其他"] += 1
        
        rows.append({
            "文件名": f.name,
            "类型": ftype,
            "大小_MB": round(size_mb, 1),
            "页数": pages if pages else "",
            "处理策略": note
        })
        
        # 终端实时输出（只打关键信息）
        flag = "√" if "文字" in ftype or "纯文本" in ftype else \
               "!" if "图文" in ftype or "Word" in ftype or "Excel" in ftype else \
               "×" if "扫描" in ftype or "图片" in ftype else "?"
        print(f"{flag} {f.name[:50]:<50} | {ftype:<10} | {size_mb:.1f}MB")
    
    # 写CSV
    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["文件名", "类型", "大小_MB", "页数", "处理策略"])
        writer.writeheader()
        writer.writerows(rows)
    
    # 汇总
    print("-" * 70)
    print("\n汇总:")
    total = len([r for r in rows if r])
    easy = summary["文字版PDF"] + summary.get("纯文本", 0)
    ocr = summary["扫描版PDF"] + summary["图片"]
    medium = summary["图文混排PDF"] + summary["Word"] + summary["PPT"] + summary["Excel"]
    
    print(f"  可直接处理: {easy} 个")
    print(f"  需解析器(docx/pptx/xlsx): {medium} 个")
    print(f"  需OCR/多模态: {ocr} 个")
    print(f"  压缩包/其他: {summary['压缩包'] + summary['其他']} 个")
    print(f"  损坏: {summary['损坏']} 个")
    print(f"\n建议: 可直接处理的 {easy} 个，快速出demo；扫描版 {ocr} 个后续可能需要多模态。")
    print(f"详细清单: {OUTPUT}")


if __name__ == "__main__":
    audit()