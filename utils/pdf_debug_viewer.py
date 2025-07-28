import fitz  # PyMuPDF
import re
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import difflib


# ----------------------------- 文本预处理工具 -----------------------------
def clean_zero_width_chars(text: str) -> str:
    """
    清除 PDF 中不可见空格字符
    """
    return (
        text.replace("\u200B", "")   # ZWSP
            .replace("\u00A0", " ")  # NBSP
            .replace("\u2002", " ")  # EN space
            .replace("\u2003", " ")  # EM space
            .replace("\u202F", " ")  # Narrow NBSP
    )


def normalize_paragraphs(text: str) -> str:
    """
    合并段落（空行视为段落分隔）
    """
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
    """
    清理页码、页脚提示、重复页眉等
    """
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


# ----------------------------- 文本统计函数 -----------------------------
def get_text_stats(text: str) -> str:
    lines = text.splitlines()
    paragraphs = text.split("\n\n")
    return f"字符数: {len(text)} | 行数: {len(lines)} | 段落数: {len(paragraphs)} | 空行: {text.count(chr(10)+chr(10))}"


# ----------------------------- 提取每页每阶段处理结果 -----------------------------
def extract_text_from_pdf_debug(pdf_path: Path) -> dict:
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
    return debug_info


# ----------------------------- 差异高亮 HTML -----------------------------
def generate_diff_html(a: str, b: str) -> str:
    """
    使用 difflib 生成 HTML 差异预览
    """
    differ = difflib.HtmlDiff()
    return differ.make_file(a.splitlines(), b.splitlines(), fromdesc="前一步", todesc="当前步骤", context=True)


# ----------------------------- 导出所有处理文本 -----------------------------
def save_all_outputs(debug_result: dict, output_dir: Path):
    """
    将所有处理阶段文本导出为 .txt 文件
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for step, pages in debug_result.items():
        for i, page in enumerate(pages):
            out_path = output_dir / f"{step}_page_{i+1}.txt"
            out_path.write_text(page, encoding='utf-8')
    messagebox.showinfo("导出成功", f"所有文本已导出至:\n{output_dir}")


# ----------------------------- 主 GUI 界面 -----------------------------
def display_debug_gui(debug_result: dict):
    root = tk.Tk()
    root.title("📄 PDF 处理调试与导出工具")
    root.geometry("1200x800")

    steps = list(debug_result.keys())
    total_pages = len(debug_result["raw_text"])
    current_page = tk.IntVar(value=0)

    def render_page():
        for widget in content_frame.winfo_children():
            widget.destroy()

        page_idx = current_page.get()
        for step_idx, step in enumerate(steps):
            page_text = debug_result[step][page_idx]
            stats = get_text_stats(page_text)

            # 阶段标题 + 统计
            tk.Label(content_frame, text=f"【{step}】第 {page_idx + 1} 页（{stats}）",
                     font=("Arial", 14, "bold")).pack(pady=(15, 5))

            # 文本展示框
            text_widget = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, font=("Courier", 10), height=15)
            text_widget.insert(tk.END, page_text[:20000])
            text_widget.pack(fill="both", expand=True, padx=10, pady=5)

            # 差异对比按钮（从第二阶段起）
            if step_idx > 0:
                html_diff = generate_diff_html(debug_result[steps[step_idx - 1]][page_idx], page_text)
                btn = tk.Button(content_frame, text=f"查看 [{steps[step_idx - 1]} → {step}] 差异HTML",
                                command=lambda h=html_diff: show_diff_window(h))
                btn.pack(pady=5)

    def show_diff_window(html: str):
        """
        生成 HTML 差异文件并用浏览器打开
        """
        path = Path("diff_preview.html")
        path.write_text(html, encoding="utf-8")
        import webbrowser
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
            save_all_outputs(debug_result, Path(out_dir))

    # 顶部按钮行
    top_frame = tk.Frame(root)
    top_frame.pack(fill="x", pady=10)
    tk.Button(top_frame, text="上一页", command=prev_page).pack(side="left", padx=10)
    tk.Button(top_frame, text="下一页", command=next_page).pack(side="left", padx=10)
    tk.Button(top_frame, text="导出所有阶段为TXT", command=save_all).pack(side="right", padx=10)

    # 内容展示区
    content_frame = tk.Frame(root)
    content_frame.pack(fill="both", expand=True)
    render_page()

    root.mainloop()


# ----------------------------- 脚本入口 -----------------------------
if __name__ == "__main__":
    pdf_file = Path("../utils/move1.pdf")  # ← 修改为你的文件路径
    if pdf_file.exists():
        result = extract_text_from_pdf_debug(pdf_file)
        display_debug_gui(result)
    else:
        print(f"[!] 文件未找到: {pdf_file}")
