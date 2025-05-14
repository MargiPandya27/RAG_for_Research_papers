# main.py
from datasets import load_dataset
from vector_db_processing import DocumentProcessor
import google.generativeai as genai

# Load dataset
data_arxiv = load_dataset("ccdv/arxiv-summarization")["train"]

# Select a subset of the dataset (example: first 2 documents)
sample_data = data_arxiv.select(range(2))

# Instantiate the processor
processor = DocumentProcessor(dataset=sample_data, output_dir="output")

# Process the dataset
processed_documents = processor.process_dataset()

genai.configure(api_key="YOUR_API_KEY")
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
