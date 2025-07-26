# Retrieval-Augmented Generation (RAG) for Research Papers

A comprehensive **Retrieval-Augmented Generation (RAG)** system designed to search, analyze, and extract insights from academic research papers. This project provides intelligent document processing, semantic search, and AI-powered question answering capabilities for research paper collections.

![Model Architecture](https://github.com/MargiPandya27/RAG_for_Research_Papers/blob/main/diagram.png)

## Features

- **Multi-Modal Document Processing**: Support for both predefined corpora and user-uploaded PDF papers
- **Advanced Text Processing**: Intelligent section extraction, chunking, and metadata extraction
- **Semantic Search**: SciBERT embeddings with FAISS HNSW indexing for efficient similarity search
- **AI-Powered Q&A**: Integration with Google Gemini for intelligent question answering
- **Metadata Extraction**: Automatic extraction of paper metadata including authors, abstracts, and citations
- **Flexible Architecture**: Modular design supporting different use cases and data sources

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RAG_for_Research_Papers-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys** (optional, for Gemini integration):
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

## Quick Start

### Option 1: Process Your Own PDF Papers

1. **Place your PDF files** in the `pdf_folder/pdfs/` directory

2. **Run the processing pipeline**:
   ```bash
   cd pdf_folder
   python main.py
   ```

3. **Query your papers**:
   ```python
   # The system will automatically process your PDFs and create embeddings
   # You can then ask questions about your papers
   ```

### Option 2: Use Predefined Corpus

1. **Run with arxiv dataset**:
   ```bash
   cd corpus
   python main.py
   ```

2. **Query the corpus**:
   ```python
   # The system will load the arxiv dataset and allow you to query it
   ```

## Project Structure

```
RAG_for_Research_Papers-main/
├── corpus/                          # Predefined corpus processing
│   ├── main.py                      # Main script for corpus processing
│   ├── embeddings.py                # Embedding generation
│   ├── faiss_indexing.py           # FAISS index creation
│   ├── text_processing.py          # Text preprocessing utilities
│   ├── vector_db_processing.py     # Vector database operations
│   └── output/                     # Processed chunks output
├── pdf_folder/                      # User PDF processing
│   ├── main.py                      # Main script for PDF processing
│   ├── embedding.py                 # SciBERT embeddings
│   ├── faiss_index.py              # FAISS indexing implementation
│   ├── pdf_processing.py           # PDF text extraction
│   ├── text_utils.py               # Text utilities
│   ├── pdfs/                       # User PDF files
│   ├── processed_chunks_json/      # Processed text chunks
│   └── processed_papers.json       # Extracted paper data
├── meta_data/                       # Metadata extraction
│   ├── main.py                      # Metadata extraction script
│   ├── extract_metadata.py         # PDF metadata extraction
│   ├── embedding.py                # Metadata embeddings
│   ├── retrieve.py                 # Metadata retrieval
│   ├── pdfs/                       # PDF files for metadata
│   └── pdf_metadata_output.json    # Extracted metadata
├── requirements.txt                 # Python dependencies
├── diagram.svg                      # System architecture diagram
└── README.md                       # This file
```

## Usage Examples

### Basic Query Example

```python
# After processing your papers, you can query them like this:
query = "What are the main findings about transformer architectures?"

# The system will:
# 1. Generate embeddings for your query
# 2. Find the most relevant document chunks
# 3. Generate an AI-powered response
```

### Advanced Usage

```python
from embedding import SciBERTEmbeddings
from faiss_index import DocumentProcessor

# Initialize the embedding model
embedder = SciBERTEmbeddings()

# Process documents
processor = DocumentProcessor(
    dataset=your_papers,
    output_dir="processed_chunks"
)

# Create embeddings and index
processed_documents = processor.process_dataset()

# Search for relevant documents
query_embedding = embedder.embed_query("Your question here")
distances, indices = processor.search_faiss_index(query_embedding, k=5)
```

## Architecture

The system follows a modular architecture with the following key components:

### 1. Document Processing Pipeline
- **PDF Extraction**: PyMuPDF-based text extraction with intelligent section detection
- **Text Chunking**: Semantic chunking with configurable overlap
- **Metadata Extraction**: Automatic extraction of paper metadata

### 2. Embedding System
- **SciBERT Model**: Domain-specific embeddings for scientific text
- **Vector Generation**: Efficient embedding generation for documents and queries

### 3. Search & Retrieval
- **FAISS HNSW Index**: High-performance similarity search
- **Semantic Matching**: Context-aware document retrieval
- **Ranking**: Distance-based relevance scoring

### 4. AI Integration
- **Google Gemini**: Large language model for answer generation
- **Context Assembly**: Intelligent context preparation for LLM
- **Response Generation**: Structured answer generation

## ⚙️ Configuration

### Environment Variables

```bash
# Google Gemini API Key (optional)
export GOOGLE_API_KEY="your-api-key"

# CUDA Configuration (for GPU acceleration)
export KMP_DUPLICATE_LIB_OK="TRUE"
```

### Model Configuration

The system uses the following default models:
- **Embedding Model**: `allenai/scibert_scivocab_uncased`
- **LLM**: `gemini-2.0-flash`
- **FAISS Index**: HNSW with 768 dimensions

## API Reference

### DocumentProcessor

Main class for processing documents and creating search indices.

```python
class DocumentProcessor:
    def __init__(self, dataset, output_dir, index_name="research_papers_index")
    def process_dataset(self)
    def search_faiss_index(self, query_embedding, k=5)
```

### SciBERTEmbeddings

SciBERT-based embedding generation.

```python
class SciBERTEmbeddings:
    def __init__(self, model_name="allenai/scibert_scivocab_uncased")
    def embed_documents(self, texts)
    def embed_query(self, text)
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **SciBERT**: For domain-specific embeddings
- **FAISS**: For efficient similarity search
- **Google Gemini**: For AI-powered question answering
- **PyMuPDF**: For PDF text extraction

## Support

If you encounter any issues or have questions, please:

1. Check the existing issues
2. Create a new issue with detailed information
3. Include error messages and system information

---

**Note**: This project is designed for research and educational purposes. Please ensure you have the necessary permissions when processing copyrighted materials.

