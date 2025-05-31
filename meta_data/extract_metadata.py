import os
import fitz  # PyMuPDF
import re
import logging
import json
from datetime import datetime, timedelta, timezone

def decode_pdf_date(pdf_date_str):
    """Decode PDF date string to ISO format."""
    if not pdf_date_str or not isinstance(pdf_date_str, str):
        return None
    date_str = pdf_date_str.strip()
    if date_str.startswith('D:'):
        date_str = date_str[2:]
    pattern = re.compile(
        r"^(?P<year>\d{4})"
        r"(?P<month>\d{2})?"
        r"(?P<day>\d{2})?"
        r"(?P<hour>\d{2})?"
        r"(?P<minute>\d{2})?"
        r"(?P<second>\d{2})?"
        r"(?P<tz>Z|[+\-]\d{2}'?\d{2}'?)?$"
    )
    match = pattern.match(date_str)
    if not match:
        return None
    parts = match.groupdict(default='01')
    try:
        dt = datetime(
            int(parts['year']),
            int(parts['month']),
            int(parts['day']),
            int(parts['hour']),
            int(parts['minute']),
            int(parts['second'])
        )
    except Exception:
        return None
    tz = parts['tz']
    if tz == 'Z':
        dt = dt.replace(tzinfo=timezone.utc)
    elif tz and (tz.startswith('+') or tz.startswith('-')):
        sign = 1 if tz[0] == '+' else -1
        tz_clean = tz.replace("'", "")
        tz_hours = int(tz_clean[1:3])
        tz_minutes = int(tz_clean[3:5]) if len(tz_clean) > 3 else 0
        offset = timedelta(hours=tz_hours, minutes=tz_minutes)
        dt = dt.replace(tzinfo=timezone(sign * offset))
    return dt

def extract_title_from_first_page(page):
    """
    Extracts the title from the first page by collecting consecutive lines
    before 'ABSTRACT', skipping likely author and affiliation lines.
    Handles multi-line and hyphenated titles.
    """
    lines = page.get_text().split('\n')
    title_lines = []
    found_title = False
    for idx, line in enumerate(lines):
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
            # Stop collecting if we hit a short or unrelated line after starting title
            break
    # Merge lines, remove hyphenation at line breaks
    title = ' '.join(title_lines)
    title = re.sub(r'-\s+', '', title)  # Remove hyphens at end of lines
    title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
    return title.strip() if title else None

def extract_metadata_from_pdf(pdf_path):
    metadata = {
        'file_name': os.path.basename(pdf_path),
        'title': None,
        'authors': None,
        'keywords': None,
        'creation_date': None,
        'text_snippet': None,
        'abstract': None,
    }
    try:
        with fitz.open(pdf_path) as doc:
            doc_metadata = doc.metadata
            # Use None if metadata is missing or empty
            metadata['title'] = doc_metadata.get('title') or None
            metadata['authors'] = doc_metadata.get('author') or None
            metadata['keywords'] = doc_metadata.get('keywords') or None
            dt = decode_pdf_date(doc_metadata.get('creationDate'))
            metadata['creation_date'] = dt.isoformat() if dt else None

            full_text = ""
            for page in doc:
                full_text += page.get_text()
            metadata['text_snippet'] = full_text[:500]

            lines = full_text.split('\n')

            # Title heuristic fallback
            if not metadata['title']:
                title = extract_title_from_first_page(doc[0])
                if title:
                    metadata['title'] = title

            # Authors heuristic: look for lines with email or typical author patterns
            if not metadata['authors']:
                author_lines = [line for line in lines[:15] if '@' in line or re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+', line)]
                if author_lines:
                    metadata['authors'] = "; ".join(author_lines)

            # Keywords heuristic: look for a line starting with "Keywords"
            if not metadata['keywords']:
                for line in lines:
                    if line.lower().startswith("keywords"):
                        metadata['keywords'] = line.split(":", 1)[-1].strip()
                        break

            # Abstract extraction
            abstract = None
            abstract_pattern = re.compile(r'^\s*(abstract|\\section\{abstract\}|\\begin\{abstract\}|abstract\.)\s*[:.]?\s*$', re.IGNORECASE)
            for i, line in enumerate(lines):
                if abstract_pattern.match(line):
                    abstract_lines = []
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if (re.match(r'^(keywords|introduction|1\.|\\section|\\begin)', next_line, re.IGNORECASE) and len(abstract_lines) > 2) or len(" ".join(abstract_lines)) > 2000:
                            break
                        if next_line:
                            abstract_lines.append(next_line)
                    abstract = " ".join(abstract_lines).strip()
                    break
            metadata['abstract'] = abstract if abstract else None

    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")

    return metadata

# # Example usage for a folder of PDFs
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     pdf_folder = "pdfs"  # Change to your folder path
#     results = []
#     for file in os.listdir(pdf_folder):
#         if file.lower().endswith(".pdf"):
#             pdf_path = os.path.join(pdf_folder, file)
#             meta = extract_metadata_from_pdf(pdf_path)
#             results.append(meta)
#     # Save to JSON
#     with open("metadata_output.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)
#     print("Extraction complete! Results saved to metadata_output.json")
