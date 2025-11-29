# Architecture Diagram Explanation

## Overview
The `architecture_diagram.png` illustrates the complete end-to-end flow of the Knowledge Base Agent, a Retrieval-Augmented Generation (RAG) system built with Streamlit, ChromaDB, and Google Gemini API.

## System Layers

### 1. User Interface Layer
- **Streamlit Web UI** (`app.py`): Main application entry point providing interactive web interface
- **File Upload**: Accepts PDF, DOCX, and TXT documents for ingestion
- **Question Input**: Natural language query interface for users
- **Answer Display**: Renders AI-generated answers with source citations

### 2. Application Core
- **Document Ingestion**: Processes uploaded files through the pipeline
- **Text Chunking**: Splits documents into 1200-character chunks with 80-character overlap for optimal retrieval
- **Query Processing**: Handles user questions and orchestrates retrieval + generation
- **Semantic Retrieval**: Performs vector similarity search to find Top-5 most relevant chunks (threshold: 0.8)
- **Prompt Engineering**: Constructs prompts combining retrieved context with user questions

### 3. Utility Functions (`utils.py`)
- **Document Parsers**: Extracts text from PDF (pdfplumber), DOCX (python-docx), and TXT files
- **gemini_embed()**: Generates 768-dimensional embeddings with parallel workers (default: 4 threads)
- **gemini_chat()**: Sends prompts to Gemini LLM and returns generated answers

### 4. External Services (Google Gemini API)
- **text-embedding-004**: Produces dense vector representations of text; includes retry logic with exponential backoff
- **gemini-2.0-flash**: Chat completion model for context-aware answer generation

### 5. Persistent Storage
- **ChromaDB**: Local vector database using cosine similarity for semantic search
- **Metadata Store**: Tracks chunk IDs, source filenames, and text previews

### 6. Configuration
- **.env file**: Stores API keys and performance tuning parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBED_WORKERS`)

## Data Flows

### Ingestion Flow (Document → Database)
1. User uploads a document via the UI
2. Document parser extracts raw text
3. Text is split into overlapping chunks (1200 chars each)
4. Chunks are sent to Gemini API for embedding (parallel processing with 4 workers)
5. Resulting vectors and metadata are stored in ChromaDB

### Query Flow (Question → Answer)
1. User submits a natural language question
2. Question is embedded using Gemini API
3. Query vector is used to search ChromaDB for similar chunks
4. Top-5 most relevant chunks are retrieved (based on cosine similarity)
5. Retrieved chunks + question are combined into a prompt
6. Prompt is sent to Gemini LLM for answer generation
7. Generated answer is displayed with source snippets and similarity scores

## Key Features & Optimizations

- **Parallel Embedding**: 4 concurrent workers reduce ingestion time by ~4x compared to sequential processing
- **Optimized Chunking**: Larger chunks (1200 vs 800) reduce API calls by ~30% while maintaining context quality
- **Persistent Storage**: ChromaDB enables fast vector search (sub-second on 1000s of chunks)
- **Retry Logic**: Exponential backoff handles rate limits and transient failures gracefully
- **Source Attribution**: Every answer cites specific document chunks to prevent hallucination
- **Clean UI**: Custom Streamlit styling with dark theme and contrasting components

## Technology Stack

| Component       | Technology                              |
|-----------------|-----------------------------------------|
| Frontend        | Streamlit 1.51.0                        |
| Vector Database | ChromaDB 1.3.5 (cosine distance)        |
| Embeddings      | Google Gemini text-embedding-004        |
| LLM             | Google Gemini 2.0 Flash                 |
| Parsing         | pdfplumber, python-docx                 |
| Language        | Python 3.10+                            |

## Performance Characteristics

- **Parallel embedding**: ~4x faster than sequential (4 workers vs 1)
- **Larger chunks**: ~30% reduction in API calls (1200 vs 800 chars)
- **Vector search**: Sub-second retrieval even with thousands of indexed chunks
- **End-to-end query latency**: Typically 3-5 seconds (embed query → search → generate answer)

## Architecture Decisions

### Why ChromaDB?
- **Local-first**: No cloud dependency, fast disk-based persistence
- **Cosine similarity**: Industry-standard for semantic search
- **Simple API**: Easy to integrate and maintain

### Why Parallel Embedding?
- **Bottleneck**: Embedding generation is I/O-bound (network latency to Gemini API)
- **Solution**: ThreadPoolExecutor with 4 workers saturates network bandwidth without hitting rate limits
- **Trade-off**: More workers = faster but higher risk of 429 errors; 4 is empirically optimal

### Why 1200-character Chunks?
- **Context window**: Balances information density vs retrieval precision
- **Overlap (80 chars)**: Ensures sentences aren't split across chunks
- **API efficiency**: Fewer chunks = fewer embedding calls = faster indexing

### Why Gemini API?
- **Cost-effective**: Free tier suitable for demos and small-scale use
- **Performance**: text-embedding-004 produces high-quality 768-dim vectors
- **Integration**: REST API with retry logic avoids SDK complexity

## File Organization

```
project/
├── app.py                  # Main Streamlit application
├── utils.py                # Core utilities (embed, chat, parse)
├── ingest.py               # Document ingestion logic
├── docs/
│   └── architecture_diagram.png   # This diagram
├── knowledge_base/         # Sample documents for testing
├── tests/                  # Unit tests (embeddings, utils)
├── requirements.txt        # Python dependencies
├── .env.example            # Configuration template
└── README.md               # Project documentation
```

## Future Enhancements

- **Streaming responses**: Display answer tokens as they generate (reduce perceived latency)
- **Background ingestion**: Queue-based indexing to avoid UI blocking
- **OCR support**: Extract text from scanned PDFs and images
- **Cloud vector store**: Enable multi-user sessions with hosted ChromaDB or Pinecone
- **Evaluation suite**: Measure faithfulness, answer quality, and retrieval recall
