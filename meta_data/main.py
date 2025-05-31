
import os
import json
from extract_metadata import extract_metadata_from_pdf
from embedding import embedding
from sentence_transformers import SentenceTransformer
from retrieve import retrieve
import google.generativeai as genai


# Main execution
pdf_folder = "pdfs"
output_json_path = "pdf_metadata_output.json"
results = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file)
        metadata = extract_metadata_from_pdf(pdf_path)
        results.append(metadata)

# Save results to JSON file
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Metadata saved to {output_json_path}")


# Load JSON file into Python list of dicts
with open("pdf_metadata_output.json", "r", encoding="utf-8") as f:
    documents = json.load(f)  # documents is now a list of dicts

# Now call your embedding function
index = embedding(documents)
print('index', index)

genai.configure(api_key="AIzaSyDgXKQojjiZqdH468J6cP_ZZGedC49RGT4")
llm = genai.GenerativeModel("gemini-2.0-flash")


query = "Can you give me email address of the authors of Memory-Efficient Fine-Tuning of Transformers via Token Selection?"

# Step 1: Retrieval
model = SentenceTransformer("all-MiniLM-L6-v2")
top_docs = retrieve(query, index, documents, model)

prompt = f"""
    Go through all the research papers given as corpus and their relevant metadata as given below:

    {top_docs}
    Please answer the {query} with help of the given in the context.
    """


# Step 2: Generation
response = llm.generate_content(prompt)
print("Generated Response: ", response.text)


