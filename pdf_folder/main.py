import json
import google.generativeai as genai
from embedding import SciBERTEmbeddings
from faiss_index import DocumentProcessor
from pdf_processing import process_pdf_folder
from pathlib import Path
import google.generativeai as genai


def save_as_json(data: dict, output_path: str) -> None:
    """Save processed data to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved output to {output_path}")
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")


def load_and_convert_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    dataset = []
    for paper_id, paper in json_data.items():
        full_text = " ".join([sec["content"] for sec in paper["sections"]])
        section_names = [sec["section"] for sec in paper["sections"]]
        dataset.append({
            "title": paper["title"],
            "article": full_text,
            "section_names": section_names
        })

    return dataset

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # [2][5]


folder_path = "pdfs"
output_json = "processed_papers.json"  # Output filename

# Process papers and save to JSON
formatted_output = process_pdf_folder(folder_path)
save_as_json(formatted_output, output_json)

json_path = "processed_papers.json"  # <-- Update this with your file
dataset = load_and_convert_json(json_path)

processor = DocumentProcessor(
    dataset=dataset,
    output_dir="processed_chunks_json"
)
processed_documents = processor.process_dataset()




genai.configure(api_key="AIzaSyD5ilxVx0M1YyNomVnabEDwEiif1D29es0")

model = genai.GenerativeModel("gemini-2.0-flash")


# Query for document retrieval
query = "Summarize what is segmentation?"

# Get the embedding for the query
query_embedding = processor.embedder.embed_query(query)

# Perform the FAISS search
distances, indices = processor.search_faiss_index(query_embedding, k=10)

# Check if any documents were retrieved
if len(indices[0]) == 0:
    print("No relevant documents found.")
else:
    # Join the page content from the retrieved documents to form a larger context
    context = "\n\n".join([processed_documents[idx].page_content for idx in indices[0]])

    # Optional: If the context is too long, you might want to truncate it
    def truncate_context(context, max_length=3000):
        if len(context) > max_length:
            context = context[:max_length]
        return context

    # Truncate context if necessary
    #context = truncate_context(context)

    # Craft the prompt to guide the model in summarizing the related work
    prompt = f"""
    You are reading a research paper. The following is extracted content from the abstract and related work sections:

    {context}
    Please answer the {query} with help of the given in the context in a well structured manner
    """

    # Generate the response from the model
    try:
        # Assuming `model` is a language model that can generate text based on the prompt
        response = model.generate_content(prompt)
        print("Generated Response: ", response.text)
    except Exception as e:
        print(f"Error generating response: {e}")
