import os
import re
import fitz  # PyMuPDF
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import quote


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    提取 PDF 文本并进行清洗：
    - 去除页码、页眉、页脚；
    - 去除零宽字符等不可见字符；
    """
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        text = page.get_text()
        text = clean_page_artifacts(text)
        text = clean_zero_width_chars(text)
        all_text.append(text.strip())
    doc.close()
    return "\n".join(all_text)


def clean_zero_width_chars(text: str) -> str:
    """
    清除零宽空格、NBSP、EMSP 等不可见或特殊空格字符
    """
    # 替换为普通空格
    text = text.replace("\u200B", "")     # ZWSP
    text = text.replace("\u00A0", " ")    # NBSP
    text = text.replace("\u2002", " ")    # EN space
    text = text.replace("\u2003", " ")    # EM space
    text = text.replace("\u202F", " ")    # NARROW NBSP
    return text


def clean_page_artifacts(text: str) -> str:
    # 去除典型页码、页眉页脚内容
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
    """
    基于章节编号（如 2, 2.1, 2.2）来提取整个文献综述块。
    提取从 section_number 开始的段落，直到下一个非该前缀的章节。
    """
    pattern = rf"\n\s*{section_number}(\.\d+)*\.?\s+[^\n]+\n"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return "", text
    start = matches[0].start()

    # 找到第一个不是以 section_number 开头的下一个章节位置
    end = None
    for i in range(1, len(matches)):
        if not matches[i].group().strip().startswith(section_number):
            end = matches[i].start()
            break
    if not end:
        # 或尝试找“3.”、“4.”等新章节
        next_section = re.search(rf"\n\s*[3-9](\.\d+)*\.?\s+[A-Z][^\n]+\n", text[matches[0].end():])
        end = matches[0].end() + (next_section.start() if next_section else 0)

    review = text[start:end].strip()
    rest = (text[:start] + text[end:]).strip() if end else text[:start]
    return review, rest


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


def query_crossref_apa(citation: str) -> Optional[str]:
    """
    使用 CrossRef 查询 citation，并返回 APA 风格引用。
    兼容异常格式、网络问题，支持代理配置。
    """

    # 可选：根据你的 VPN 设置调整代理
    proxies = {
         "http": "http://127.0.0.1:7890",   # 如果你使用 VPN 代理，请取消注释
         "https": "http://127.0.0.1:7890"
    }

    # 1. 格式匹配校验
    match = re.match(r"(.*),\s*(\d{4}[a-z]?)", citation)
    if not match:
        print(f"[!] 引用格式无法解析，跳过: {citation}")
        return None

    try:
        author_part, year = match.groups()
        author_query = author_part.strip().split(";")[0].strip()

        # 2. 拼接 URL
        query = quote(f"{author_query} {year}")
        url = f"https://api.crossref.org/works?query={query}&rows=1"

        # 3. 发起请求，增加超时时间，支持代理
        resp = requests.get(url, timeout=20, proxies=proxies or None)
        resp.raise_for_status()
        data = resp.json()

        # 4. 解析结果
        if data["message"]["items"]:
            item = data["message"]["items"][0]
            authors = item.get("author", [])
            author_str = ", ".join(f"{a['family']}, {a.get('given', '')}" for a in authors[:3]) if authors else "Unknown"
            year_str = item.get("issued", {}).get("date-parts", [[year]])[0][0]
            title = item.get("title", [""])[0]
            journal = item.get("container-title", [""])[0]
            doi = item.get("DOI", "")
            apa = f"{author_str} ({year_str}). {title}. {journal}. https://doi.org/{doi}"
            return apa

        else:
            print(f"[!] 未找到 CrossRef 数据: {citation}")
    except Exception as e:
        print(f"[!] CrossRef 查询失败: {citation} - {e}")

    return None


def enrich_references(citations: list[str], output_dir: Path, paper_stem: str):
    meta_dir = output_dir / f"{paper_stem}_citations"
    meta_dir.mkdir(exist_ok=True)
    for citation in citations:
        apa = query_crossref_apa(citation)
        out_path = meta_dir / f"{sanitize_filename(citation)}.txt"
        if apa:
            save_txt(apa, out_path)
        else:
            save_txt(f"[未找到元信息] {citation}", out_path)


def process_pdf(pdf_path: Path, output_dir: Path):
    print(f"[+] 正在处理: {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)
    print("[DEBUG] 文本预览:", text[:500])

    review_text, maintext_wo_review = extract_review_by_section_number(text, section_number="2")
    print("[DEBUG] Review内容预览:", review_text[:500])

    references_in_review = extract_references_in_review(review_text)
    print("[DEBUG] 引文预览:", references_in_review)

    review_structure = extract_review_structure(review_text)
    print("[DEBUG] 结构提取:", review_structure)

    all_references = extract_all_references(text)
    print("[DEBUG] 全部引用预览:", all_references[:500])

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
    input_dir = Path("../utils")         # 替换为你自己的输入目录
    output_dir = Path("")      # 替换为你希望的输出目录
    batch_process(input_dir, output_dir)
