# pdf_processing.py
import os
import re
import pymupdf as fitz  # PyMuPDF for text extraction
import json

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF with error handling."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {str(e)}")
        return ""

def chunk_by_sections_with_titles(text: str, section_titles: list = None) -> tuple:
    """
    Splits one paper into section chunks while ignoring reference sections.
    Returns title and list of section chunks.
    """
    # Default sections to look for
    if section_titles is None:
        section_titles = ["Abstract", "Introduction", "Methods",
                         "Results", "Discussion", "Conclusion"]

    # Sections to explicitly ignore
    ignore_sections = ["References", "Bibliography", "Acknowledgments"]

    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

    if not lines:
        return "Untitled", []

    # Improved title detection (first line with proper capitalization)
    title = next((line for line in lines if line.istitle()), "Untitled")
    content_lines = lines[1:]

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
            section_name = match.group(3)  # Get the main section name
            full_match = match.group(1)

            # Check if we're entering an ignored section
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

    # Add final section if valid
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
            print(f"Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            if text:
                papers.append(text)
    return format_papers(papers)

# Add this function to handle JSON saving
def save_as_json(data: dict, output_path: str) -> None:
    """Save processed data to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved output to {output_path}")
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")
        

# Modified main section
if __name__ == "__main__":
    folder_path = "pdfs"
    output_json = "processed_papers.json"  # Output filename

    # Process papers and save to JSON
    formatted_output = process_pdf_folder(folder_path)
    save_as_json(formatted_output, output_json)