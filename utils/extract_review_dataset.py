import os
import re
import fitz  # PyMuPDF
import requests
from pathlib import Path
from typing import Optional


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    从 PDF 文件中提取文本，增强清理逻辑：去除页码、脚注、ZWSP等。
    """
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        text = page.get_text()
        text = clean_page_artifacts(text)
        text = clean_zero_width_chars(text)
        all_text.append(text.strip())
    return "\n".join(all_text)


def clean_zero_width_chars(text: str) -> str:
    """
    清除零宽空格 (ZWSP) 以及其他不可见字符
    """
    return text.replace("\u200B", "")


def clean_page_artifacts(text: str) -> str:
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


def extract_review_section(text: str) -> tuple[str, str]:
    """
    提取文献综述部分：兼容“Literature Review”、“Related Work”等；
    支持非编号形式标题。
    """
    patterns = [
        r"\n\s*(\d+(\.\d+)?\.?\s+Literature Review.*?)\n",
        r"\n\s*(\d+(\.\d+)?\.?\s+Related Work.*?)\n",
        r"\n\s*(Literature Review)\s*\n",
        r"\n\s*(Related Work)\s*\n"
    ]
    for pat in patterns:
        pattern = re.compile(pat, re.IGNORECASE)
        match = pattern.search(text)
        if match:
            start = match.start()
            next_section = re.search(r"\n\s*\d+(\.\d+)*\.?\s+[A-Z][^\n]{3,80}\n", text[match.end():])
            end = match.end() + (next_section.start() if next_section else 0)
            review = text[match.end():end].strip()
            rest = (text[:start] + text[end:]).strip()
            return review, rest
    return "", text


def extract_review_structure(review_text: str) -> str:
    structure_lines = []
    for line in review_text.splitlines():
        if re.match(r"^\s*(\d+(\.\d+)*|\.\d+)\s+[A-Z][\w\- ]+", line.strip()):
            structure_lines.append(line.strip())
    return "\n".join(structure_lines)


def extract_all_references(text: str) -> str:
    match = re.search(r"(?i)\n\s*References\s*\n", text)
    if not match:
        return ""
    refs_text = text[match.end():].strip()
    return refs_text


def extract_references_in_review(review_text: str) -> list[str]:
    citations = set(re.findall(r"\(([^()]*?\d{4}[a-z]?)\)", review_text))
    return sorted(citations)


def save_txt(content: str, path: Path):
    path.write_text(content.strip(), encoding='utf-8')


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def query_semantic_scholar(citation: str) -> Optional[dict]:
    query = re.sub(r"[\(\)]", "", citation).strip()
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,authors,abstract&limit=1"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("data"):
            item = data["data"][0]
            title = item.get("title", "")
            authors = ", ".join([a.get("name", "") for a in item.get("authors", [])])
            abstract = item.get("abstract", "")
            return {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "citation": citation
            }
    except Exception as e:
        print(f"[!] Semantic Scholar 查询失败: {citation} - {e}")
    return None


def enrich_references(citations: list[str], output_dir: Path, paper_stem: str):
    meta_dir = output_dir / f"{paper_stem}_citations"
    meta_dir.mkdir(exist_ok=True)
    for citation in citations:
        data = query_semantic_scholar(citation)
        if data:
            out_path = meta_dir / f"{sanitize_filename(citation)}.txt"
            content = f"""Title: {data['title']}
Authors: {data['authors']}
Abstract: {data['abstract']}
Citation: {data['citation']}"""
            save_txt(content, out_path)


def process_pdf(pdf_path: Path, output_dir: Path):
    print(f"[+] 正在处理: {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)
    review_text, maintext_wo_review = extract_review_section(text)
    references_in_review = extract_references_in_review(review_text)
    review_structure = extract_review_structure(review_text)
    all_references = extract_all_references(text)
    base_name = pdf_path.stem
    save_txt(review_text, output_dir / f"{base_name}.review.txt")
    save_txt(maintext_wo_review, output_dir / f"{base_name}.maintext_wo_review.txt")
    save_txt("\n".join(references_in_review), output_dir / f"{base_name}.references_in_review.txt")
    save_txt(review_structure, output_dir / f"{base_name}.review_structure.txt")
    save_txt(all_references, output_dir / f"{base_name}.all_references.txt")
    enrich_references(references_in_review, output_dir, base_name)
    print(f"[✓] 完成：{base_name} → 生成 5+N 个 txt 文件\n")


def batch_process(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("[!] 未发现 PDF 文件")
        return
    for pdf_file in pdf_files:
        process_pdf(pdf_file, output_dir)



if __name__ == "__main__":
    # 设置输入文件夹和输出文件夹路径
    input_dir = Path("../utils")         # 替换为你自己的 PDF 文件夹路径
    output_dir = Path("")     # 替换为你希望存放结果的目录

    batch_process(input_dir, output_dir)
