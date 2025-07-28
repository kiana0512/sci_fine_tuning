"""
PDF 文本调试与综述结构提取工具（完整版本）

功能包括：
- 分页展示每页的文本清洗阶段（原始→去页眉→清除字符→合并段落）
- 每阶段差异高亮 HTML 可视化
- 全文合并后提取文献综述段、结构目录、引用列表、参考文献等
- 独立界面展示综述结构
- 所有阶段内容支持导出 TXT
"""

import fitz  # PyMuPDF
import re
import difflib
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
import webbrowser

# ------------------ 文本处理函数 ------------------

def clean_zero_width_chars(text: str) -> str:
    """去除零宽字符、不可见空格"""
    return (
        text.replace("\u200B", "")
        .replace("\u00A0", " ")
        .replace("\u2002", " ")
        .replace("\u2003", " ")
        .replace("\u202F", " ")
    )

def normalize_paragraphs(text: str) -> str:
    """将连续文本行合并为段落"""
    lines = text.splitlines()
    paras, current = [], []
    for line in lines:
        striped = line.strip()
        if not striped:
            if current:
                paras.append(" ".join(current))
                current = []
        else:
            current.append(striped)
    if current:
        paras.append(" ".join(current))
    return "\n\n".join(paras)

def clean_page_artifacts(text: str) -> str:
    """去除页码、页脚页眉、重复内容"""
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    text = re.sub(r"(\S)\s{1,3}\d{1,4}\n", r"\1\n", text)
    text = re.sub(r"\n.*Page\s+\d{1,4}.*\n", "\n", text, flags=re.IGNORECASE)
    lines = text.splitlines()
    line_counts = {}
    for line in lines:
        key = line.strip()
        if key:
            line_counts[key] = line_counts.get(key, 0) + 1
    common_lines = {k for k, v in line_counts.items() if v > 5}
    return "\n".join([l for l in lines if l.strip() not in common_lines])

def extract_review_by_section_number(text: str, section_number: str = "2") -> tuple[str, str]:
    """基于章节编号提取文献综述段落"""
    pattern = rf"\n\s*{section_number}(\.\d+)*\.?\s+[^\n]+\n"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return "", text
    start = matches[0].start()
    end = None
    for i in range(1, len(matches)):
        if not matches[i].group().strip().startswith(section_number):
            end = matches[i].start()
            break
    if not end:
        next_section = re.search(rf"\n\s*[3-9](\.\d+)*\.?\s+[A-Z][^\n]+\n", text[matches[0].end():])
        end = matches[0].end() + (next_section.start() if next_section else 0)
    review = text[start:end].strip()
    rest = (text[:start] + text[end:]).strip() if end else text[:start]
    return review, rest

def extract_review_structure(review_text: str) -> str:
    """提取综述中的章节结构（如 2.1 XXX）"""
    structure_lines = []
    for line in review_text.splitlines():
        if re.match(r"^\s*(\d+(\.\d+)*|\.\d+)\s+[A-Z][\w\- ]+", line.strip()):
            structure_lines.append(line.strip())
    return "\n".join(structure_lines)

def extract_all_references(text: str) -> str:
    """提取全文末尾 References 段落"""
    match = re.search(r"(?i)\n\s*References\s*\n", text)
    if not match:
        return ""
    refs_text = text[match.end():].strip()
    return refs_text

def extract_references_in_review(review_text: str) -> list[str]:
    """提取综述段落中的 (Author, Year) 引文"""
    citations = set(re.findall(r"\(([^()]*?\d{4}[a-z]?)\)", review_text))
    return sorted(citations)

def get_text_stats(text: str) -> str:
    """计算文本的基本信息"""
    lines = text.splitlines()
    paragraphs = text.split("\n\n")
    return f"字符数: {len(text)} | 行数: {len(lines)} | 段落数: {len(paragraphs)} | 空行: {text.count(chr(10)+chr(10))}"

def generate_diff_html(a: str, b: str) -> str:
    """生成 HTML 差异视图"""
    differ = difflib.HtmlDiff()
    return differ.make_file(a.splitlines(), b.splitlines(), fromdesc="前一步", todesc="当前步骤", context=True)

def extract_text_from_pdf_debug_full(pdf_path: Path) -> dict:
    """提取并记录每阶段结果与全局结构"""
    doc = fitz.open(pdf_path)
    debug_info = {
        "raw_text": [],
        "after_clean_artifacts": [],
        "after_clean_chars": [],
        "after_normalize_paragraphs": [],
    }
    for page in doc:
        raw = page.get_text()
        debug_info["raw_text"].append(raw)
        cleaned_artifacts = clean_page_artifacts(raw)
        debug_info["after_clean_artifacts"].append(cleaned_artifacts)
        cleaned_chars = clean_zero_width_chars(cleaned_artifacts)
        debug_info["after_clean_chars"].append(cleaned_chars)
        normalized = normalize_paragraphs(cleaned_chars)
        debug_info["after_normalize_paragraphs"].append(normalized)
    doc.close()

    full_text = "\n".join(debug_info["after_normalize_paragraphs"])
    debug_info["full_cleaned_text"] = full_text
    review_text, _ = extract_review_by_section_number(full_text)
    debug_info["review_text"] = review_text
    debug_info["review_structure"] = extract_review_structure(review_text)
    debug_info["references_in_review"] = "\n".join(extract_references_in_review(review_text))
    debug_info["all_references"] = extract_all_references(full_text)

    return debug_info

# ------------------ 全局综述结构展示窗口 ------------------
def display_global_review_gui(debug_result: dict):
    win = tk.Toplevel()
    win.title("📘 全局处理结果：综述结构与引用信息")
    win.geometry("1200x900")

    def export_global_outputs():
        out_dir = filedialog.askdirectory(title="选择导出文件夹")
        if out_dir:
            out = Path(out_dir)
            for key in ["full_cleaned_text", "review_text", "review_structure", "references_in_review", "all_references"]:
                out_file = out / f"{key}.txt"
                out_file.write_text(debug_result[key], encoding='utf-8')
            messagebox.showinfo("导出成功", f"已导出至：{out.resolve()}")

    def show_global_diff():
        html = generate_diff_html(
            debug_result["full_cleaned_text"],
            debug_result["review_text"]
        )
        path = Path("global_diff_review.html")
        path.write_text(html, encoding="utf-8")
        webbrowser.open(path.resolve().as_uri())

    btn_frame = tk.Frame(win)
    btn_frame.pack(fill="x", pady=10)
    tk.Button(btn_frame, text="导出为 TXT 文件", command=export_global_outputs).pack(side="left", padx=10)
    tk.Button(btn_frame, text="📑 查看全文 → 综述 差异HTML", command=show_global_diff).pack(side="left", padx=10)

    content_frame = tk.Frame(win)
    content_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(content_frame)
    scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for key in ["full_cleaned_text", "review_text", "review_structure", "references_in_review", "all_references"]:
        content = debug_result.get(key, "")
        stats = f"字符数: {len(content)} | 行数: {content.count(chr(10)) + 1}"
        tk.Label(scroll_frame, text=f"【{key}】（{stats}）", font=("Arial", 14, "bold")).pack(pady=(15, 5))
        text_widget = scrolledtext.ScrolledText(scroll_frame, wrap=tk.WORD, height=15, font=("Courier", 10))
        text_widget.insert(tk.END, content)
        text_widget.configure(state="disabled")
        text_widget.pack(fill="both", expand=True, padx=10, pady=5)

# ------------------ 主分页展示 GUI ------------------
def display_debug_gui_full_with_global_button(debug_result: dict):
    root = tk.Tk()
    root.title("📄 PDF 全流程调试工具")
    root.geometry("1200x900")

    steps = [k for k in debug_result.keys() if isinstance(debug_result[k], list)]
    total_pages = len(debug_result["raw_text"])
    current_page = tk.IntVar(value=0)

    def render_page():
        for widget in content_frame.winfo_children():
            widget.destroy()

        page_idx = current_page.get()
        for step_idx, step in enumerate(steps):
            page_text = debug_result[step][page_idx]
            stats = get_text_stats(page_text)
            tk.Label(content_frame, text=f"【{step}】第 {page_idx + 1} 页（{stats}）", font=("Arial", 14, "bold")).pack(pady=(15, 5))
            text_widget = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, font=("Courier", 10), height=12)
            text_widget.insert(tk.END, page_text[:20000])
            text_widget.configure(state="disabled")
            text_widget.pack(fill="both", expand=True, padx=10, pady=5)

            if step_idx > 0:
                html_diff = generate_diff_html(debug_result[steps[step_idx - 1]][page_idx], page_text)
                btn = tk.Button(content_frame, text=f"查看 [{steps[step_idx - 1]} → {step}] 差异HTML",
                                command=lambda h=html_diff: show_diff_window(h))
                btn.pack(pady=5)

    def show_diff_window(html: str):
        path = Path("diff_preview.html")
        path.write_text(html, encoding="utf-8")
        webbrowser.open(path.resolve().as_uri())

    def prev_page():
        if current_page.get() > 0:
            current_page.set(current_page.get() - 1)
            render_page()

    def next_page():
        if current_page.get() < total_pages - 1:
            current_page.set(current_page.get() + 1)
            render_page()

    def save_all():
        out_dir = filedialog.askdirectory(title="选择导出文件夹")
        if out_dir:
            out = Path(out_dir)
            for step in steps:
                for i, page in enumerate(debug_result[step]):
                    (out / f"{step}_page_{i+1}.txt").write_text(page, encoding='utf-8')
            messagebox.showinfo("导出成功", f"所有内容已导出至：{out.resolve()}")

    top_frame = tk.Frame(root)
    top_frame.pack(fill="x", pady=10)
    tk.Button(top_frame, text="上一页", command=prev_page).pack(side="left", padx=10)
    tk.Button(top_frame, text="下一页", command=next_page).pack(side="left", padx=10)
    tk.Button(top_frame, text="📘 查看全局综述结构 →", command=lambda: display_global_review_gui(debug_result)).pack(side="left", padx=20)
    tk.Button(top_frame, text="导出全部结果为TXT", command=save_all).pack(side="right", padx=10)

    content_frame = tk.Frame(root)
    content_frame.pack(fill="both", expand=True)

    render_page()
    root.mainloop()

# ------------------ 程序入口 ------------------
if __name__ == "__main__":
    pdf_file = Path("../utils/move1.pdf")  # 修改为你的 PDF 文件路径
    if pdf_file.exists():
        result = extract_text_from_pdf_debug_full(pdf_file)
        display_debug_gui_full_with_global_button(result)
    else:
        print(f"[!] 文件未找到: {pdf_file}")