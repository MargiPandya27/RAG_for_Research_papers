import os
import re
import logging
import json
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF with error handling."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {str(e)}")
        return ""

def smart_title_extraction(lines):
    """
    Improved title extraction for academic PDFs:
    - Skips headers/footers.
    - Merges multi-line and hyphenated titles.
    - Stops at author or abstract lines.
    """
    title_lines = []
    found_title = False
    for idx, line in enumerate(lines[:15]):  # Only look at the top of the document
        clean = line.strip()
        # Stop at abstract, keywords, or introduction
        if re.match(r'(abstract|keywords|introduction)', clean, re.IGNORECASE):
            break
        # Skip empty lines and lines with emails or affiliations
        if not clean or '@' in clean or re.search(r'\b(university|institute|school|department|college|center|centre)\b', clean, re.IGNORECASE):
            continue
        # Heuristic: likely title lines are long, uppercase, or have colons
        if (clean.isupper() or ':' in clean or len(clean) > 20):
            found_title = True
            title_lines.append(clean)
        elif found_title:
            break
    # Merge lines, remove hyphenation at line breaks
    title = ' '.join(title_lines)
    title = re.sub(r'-\s+', '', title)  # Remove hyphens at end of lines
    title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
    return title.strip() if title else "Untitled"

def chunk_by_sections_with_titles(text: str, section_titles: list = None) -> tuple:
    """
    Splits one paper into section chunks while ignoring reference sections.
    Returns title and list of section chunks.
    """
    if section_titles is None:
        section_titles = ["Abstract", "Introduction", "Methods",
                         "Results", "Discussion", "Conclusion"]

    ignore_sections = ["References", "Bibliography", "Acknowledgments"]

    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        return "Untitled", []

    # Improved title extraction
    title = smart_title_extraction(lines)
    # Remove title lines from content
    title_lines_count = len(title.split('\n'))
    content_lines = lines[title_lines_count:]

    # Enhanced section pattern with numbered sections and variations
    section_pattern = re.compile(
        r'^\s*((\d+\.?\s*)?(' + '|'.join(map(re.escape, section_titles + ignore_sections)) +
        r'))\s*[:\-]?\s*',
        re.IGNORECASE
    )

    chunks = []
    current_section = None
    current_content = []
    in_ignored_section = False

    for line in content_lines:
        match = section_pattern.match(line)
        if match:
            section_name = match.group(3)
            if section_name.lower() in map(str.lower, ignore_sections):
                if current_section:
                    chunks.append((current_section, ' '.join(current_content).strip()))
                in_ignored_section = True
                current_section = None
                current_content = []
                continue
            else:
                in_ignored_section = False
                if current_section:
                    chunks.append((current_section, ' '.join(current_content).strip()))
                current_section = section_name
                current_content = [line[match.end():].strip()]
        elif not in_ignored_section:
            current_content.append(line)

    if current_section and not in_ignored_section:
        chunks.append((current_section, ' '.join(current_content).strip()))

    return title, chunks

def format_papers(papers: list, section_titles: list = None) -> dict:
    """Formats multiple papers into section chunks and returns a dictionary."""
    formatted_output = {}
    for idx, paper_text in enumerate(papers, start=1):
        title, section_chunks = chunk_by_sections_with_titles(paper_text, section_titles)
        formatted_output[f"Paper_{idx}"] = {
            "title": title,
            "sections": []
        }
        for section, content in section_chunks:
            formatted_output[f"Paper_{idx}"]["sections"].append({
                "section": section,
                "content": content
            })
    return formatted_output

def process_pdf_folder(folder_path: str) -> dict:
    """Processes all PDFs in a folder and returns a formatted JSON object."""
    papers = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            logging.info(f"Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            if text:
                papers.append(text)
            else:
                logging.warning(f"No text extracted from {filename}")
    return format_papers(papers)

def save_as_json(data: dict, output_path: str) -> None:
    """Save processed data to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved output to {output_path}")
    except Exception as e:
        logging.error(f"Error saving JSON file: {str(e)}")

if __name__ == "__main__":
    folder_path = "pdfs"
    output_json = "processed_papers.json"
    formatted_output = process_pdf_folder(folder_path)
    save_as_json(formatted_output, output_json)
