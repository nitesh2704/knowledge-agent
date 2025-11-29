# 🤖 Knowledge Base RAG Agent

A production-ready Retrieval-Augmented Generation (RAG) system that enables natural language querying of document collections. Built with Gemini API, ChromaDB vector database, and Streamlit UI.

## 📺 Demo

**Live Demo**: [`streamlit run app.py` (https://knowledge-agent-1.streamlit.app/)]

**Quick Demo GIF**: See how it works in 30 seconds
1. Upload documents (PDF/DOCX/TXT)
2. Ask questions in natural language
3. Get AI-generated answers with source citations

## 🎯 Features

- ✅ **Multi-Format Support**: PDF, DOCX, TXT document processing
- ✅ **Semantic Search**: Gemini text-embedding-004 for accurate retrieval
- ✅ **Context-Aware Answers**: Gemini 2.0 Flash LLM with retrieved context
- ✅ **Source Attribution**: Every answer cites specific document chunks
- ✅ **Live File Upload**: Add documents via UI without re-indexing
- ✅ **Database Management**: Clear/reset knowledge base with one click
- ✅ **Document Tracking**: See all indexed documents and chunk counts
- ✅ **Debug Mode**: Inspect prompts, bypass LLM, view similarity scores
- ✅ **Robust Error Handling**: Automatic retry with exponential backoff
- ✅ **Production Ready**: Comprehensive testing and diagnostic tools

# Knowledge Base Agent

## Overview
This project is a Knowledge Base Agent that answers questions using your uploaded documents. It performs retrieval-augmented generation (RAG): relevant text chunks are retrieved from a local vector store and used to generate a grounded answer with the Gemini API, presented in a clean Streamlit UI.

## Features & Limitations
- Features: multi-format ingestion (PDF, DOCX, TXT); local persistent vector store; fast retrieval; concise answers grounded in retrieved context; upload-and-index from the UI; ability to clear/reset database; performance-tuned chunking and parallel embeddings.
- Limitations: quality depends on the uploaded documents; requires a Gemini API key and internet access; very large PDFs may take time to parse; retrieval is local (no cloud sync) unless configured; rate limits may apply with high parallelism.

## Tech Stack & APIs Used
- Python, Streamlit (UI)
- ChromaDB (vector database)
- Google Gemini API: `text-embedding-004` (embeddings), `gemini-2.0-flash` (generation)
- Parsing: `pdfplumber` (PDF), `python-docx` (DOCX)

## Setup & Run Instructions
Prerequisites: Python 3.10+, a Google Gemini API key.

1) Clone and enter the project directory.
2) Create and activate a virtual environment, then install dependencies:
### 3. Prepare Documents

Create a `docs/` folder and add your documents:

```powershell
3) Create `.env` with your key:
mkdir docs
# Copy your PDF, DOCX, or TXT files to the docs/ folder
```
4) Optional performance tuning via env (defaults are reasonable):

### 4. Index Documents

Run the ingestion script to process and index your documents:
5) Run the app:

```powershell
# Index all files in docs/ folder
Use the left panel to upload files and index them, then ask questions from the main panel.

## Potential Improvements
- Streaming responses and background ingestion queue
- OCR for scanned PDFs and image-heavy documents
- Toggle for showing/hiding source snippets and similarity scores
- Centralized cloud vector store option and multi-user sessions
- Evaluation suite (faithfulness, answer quality, retrieval recall) and auto-tuning
python ingest.py

# Or index a single file
python ingest.py --file path/to/your/document.pdf
```

This creates a `chroma_db/` folder with the vector database.

### 5. Run the Application

Launch the Streamlit app:

```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 📖 Usage

### Asking Questions

1. Enter your question in the text input
2. Click the "Ask" button
3. View the generated answer and source snippets
4. Check source attribution to verify the information

### Adding New Documents

1. Use the sidebar file uploader
2. Select PDF, DOCX, or TXT files
3. Click "Re-index uploaded files"
4. Wait for processing to complete

### Example Questions

- "What is the main topic discussed in the documents?"
- "Summarize the key findings from the research papers"
- "What are the recommended best practices mentioned?"

## 🔧 Configuration

### Model Configuration

Edit `utils.py` to change models:

```python
# TODO: Update these model names if needed
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_CHAT_MODEL = "models/gemini-pro"
```

### Chunking Parameters

Edit `ingest.py` to adjust chunking:

```python
CHUNK_SIZE = 800  # Characters per chunk
OVERLAP = 100     # Overlap between chunks
```

### Retrieval Settings

Edit `app.py` to tune retrieval:

```python
TOP_K = 5                    # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7   # Minimum similarity score (0-1)
MAX_SOURCES_IN_PROMPT = 3    # Sources to include in LLM prompt
```

## 🐛 Troubleshooting

### "GEMINI_API_KEY is not set"

- Ensure your `.env` file exists and contains the API key
- Or set the environment variable before running the app
- Check that `python-dotenv` is installed

### "Knowledge base not found"

- Run `python ingest.py` to create the vector database
- Ensure the `docs/` folder contains supported files
- Check that `chroma_db/` folder was created

### API Rate Limits

- The code includes automatic retry with exponential backoff
- If you hit persistent rate limits, reduce batch sizes in `utils.py`:
  - `gemini_embed()`: Reduce `batch_size` parameter
  - Add delays between ingest.py runs

### Empty or Poor Quality Answers

- Lower `SIMILARITY_THRESHOLD` to retrieve more chunks
- Increase `CHUNK_SIZE` for more context per chunk
- Check that your documents contain relevant information
- Review source snippets to verify retrieved content

### PDF Extraction Issues

- Some PDFs may have poor text extraction (scanned images, complex layouts)
- Use OCR-processed PDFs for better results
- Try converting problematic PDFs to DOCX first

## 🔄 Swapping Providers

The codebase is designed to be provider-agnostic. To swap embedding or LLM providers:

### For Embeddings

1. Edit `gemini_embed()` in `utils.py`
2. Replace the API endpoint and request format
3. Update response parsing to extract embedding vectors

### For Chat/LLM

1. Edit `gemini_chat()` in `utils.py`
2. Replace the API endpoint and request format
3. Update response parsing to extract generated text

### Using LangChain (Optional)

To use LangChain wrappers:

```python
# Add to requirements.txt
langchain
langchain-google-genai

# Replace utils.py functions with LangChain components
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
```

## 📁 Project Structure

```
.
├── app.py                  # Streamlit UI application
├── ingest.py               # Document ingestion script
├── utils.py                # Core utilities (text processing, API calls)
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
├── README.md              # This file
├── architecture.txt       # System architecture description
├── tests/
│   └── test_utils.py      # Unit tests
├── docs/                  # Document folder (create this)
└── chroma_db/             # Vector database (created by ingest.py)
```

## 🧪 Testing

Run unit tests:

```powershell
python -m pytest tests/
```

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

Feel free to fork and modify this project for your needs. Key areas for enhancement:

- Add support for more document formats (HTML, Markdown, etc.)
- Implement conversation history
- Add document management UI (view, delete indexed docs)
- Support multiple collections/knowledge bases
- Add authentication and multi-user support

## 📞 Support

For issues related to:

- **Gemini API**: Check [Google AI Studio documentation](https://ai.google.dev/)
- **ChromaDB**: See [ChromaDB documentation](https://docs.trychroma.com/)
- **Streamlit**: Visit [Streamlit documentation](https://docs.streamlit.io/)

## 🎭 Features & Limitations

### ✅ What It Does Well
- Fast semantic search across multiple documents
- Accurate source attribution with chunk citations
- Handles documents up to 10,000 pages
- Real-time document upload and indexing
- Prevents hallucination by requiring source citations
- Detailed debugging and diagnostic tools

### ⚠️ Current Limitations
1. **Scanned PDFs**: No OCR support (text-based PDFs only)
2. **Context Window**: Limited to top 3 chunks (~2400 chars) per query
3. **No Conversation History**: Each query is independent
4. **Single Collection**: All documents in one knowledge base
5. **Local Storage Only**: No cloud sync or distributed deployment
6. **Rate Limits**: Subject to Gemini API free tier quotas

### 🚀 Potential Improvements
- Add OCR support for scanned documents
- Implement conversation history and follow-up questions
- Multi-collection support for different knowledge bases
- Hybrid search (combine keyword + semantic)
- Re-ranking model for better relevance
- Streaming responses for improved UX
- User authentication and multi-tenancy
- Export conversations to PDF/markdown
- Advanced filters (date range, document type, source)
- Integration with external APIs (web search, databases)

## ⚠️ Important Notes

- Never commit your `.env` file with real API keys to public repositories
- Monitor your API usage at [Google AI Studio](https://aistudio.google.com/)
- Always verify answers using the source snippets shown
- Performance depends on document quality and query specificity
- For production use, implement rate limiting and caching
