"""
Streamlit Knowledge Base Agent Application.
Provides a chat interface for querying documents with RAG (Retrieval-Augmented Generation).
"""

import streamlit as st
import chromadb
from pathlib import Path
import logging
import tempfile
import os
from dotenv import load_dotenv
from utils import gemini_embed, gemini_chat, load_text_from_file, split_text

# Load environment variables for debug flags
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "kb_collection"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.8  # ChromaDB cosine distance threshold (distances typically 0.6-0.8 for relevant docs)
MAX_SOURCES_IN_PROMPT = 3
MIN_SIMILARITY = 0.6  # Minimum similarity score (1 - distance/2) for relevance

# Performance tuning (override via .env if desired)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))  # Parallel embedding threads

# Debug flags (set in .env)
DEBUG_BYPASS_LLM = os.getenv("DEBUG_BYPASS_LLM", "false").lower() == "true"
DEBUG_DUMP_PROMPTS = (
    os.getenv("DEBUG_DUMP_PROMPTS", "true").lower() == "true"
)  # Default enabled

# Page config
st.set_page_config(page_title="Knowledge Base Agent", page_icon="🤖", layout="wide")

# Inject custom CSS theme for a more polished, contrast-aware UI
st.markdown(
    """
<style>
/* Global Background */
body {background: radial-gradient(circle at 20% 20%, #1b2533 0%, #0f1624 60%, #0b121b 100%) !important;}
/* Remove default padding on main container - increased for header visibility */
.block-container {padding-top: 3.5rem;}
/* Sidebar styling */
div[data-testid="stSidebar"] {background: linear-gradient(180deg,#142030 0%, #0d1622 100%) !important; border-right: 1px solid #1f2c3b;}
div[data-testid="stSidebar"] * {color: #cfdae5 !important;}
/* Typography */
h1, h2, h3, h4, h5, h6 {color:#f0f6fc !important; font-weight:600; letter-spacing:0.5px;}
p, li, label, .stMarkdown, .stText, .stCaption, span {color:#d3dde7 !important;}
/* Cards */
.kb-card {background:#1e2735; border:1px solid #2e3b4a; border-radius:14px; padding:1.1rem 1.3rem; box-shadow:0 4px 14px rgba(0,0,0,0.35);} 
.kb-card-accent {border:1px solid #ff784f;}
/* Info / success boxes override for dark theme */
div[data-testid="stNotification"] {background:#1e2735 !important; border:1px solid #2d3a48 !important;}
/* Buttons */
button[kind="primary"] {background:linear-gradient(90deg,#ff6b4a,#ff8d60) !important; color:#fff !important; font-weight:600; border-radius:10px !important;}
button {border-radius:10px !important;}
/* Expander */
details {background:#1e2735 !important; border:1px solid #2e3b4a !important; border-radius:12px !important;}
details summary {color:#f2f7fc !important; font-weight:500;}
/* Similarity metrics */
.sim-metrics {font-size:0.75rem; letter-spacing:0.5px; color:#98b1c9; margin-top:0.35rem;}
/* Divider line */
.section-divider {height:1px; background:linear-gradient(90deg,#233140,#2f455a); margin:1.8rem 0 1.2rem; border:none;}
/* Scrollbar styling */
::-webkit-scrollbar {width:10px;}
::-webkit-scrollbar-track {background:#0f1624;}
::-webkit-scrollbar-thumb {background:#2a3948; border-radius:6px;}
::-webkit-scrollbar-thumb:hover {background:#344757;}
/* Input box */
div[data-baseweb="input"] > div {background:#1e2735 !important; border:1px solid #2d3a48 !important; border-radius:10px !important;}
div[data-baseweb="input"] input {color:#f0f6fc !important;}
/* Footer */
.footer-note {text-align:center; font-size:0.75rem; color:#72879d !important; margin-top:2rem;}
</style>
""",
    unsafe_allow_html=True,
)


# Reusable card renderer
def render_card(title: str, body: str, accent: bool = False):
    border_class = "kb-card-accent" if accent else ""
    st.markdown(
        f"""
        <div class="kb-card {border_class}">
            <div style='font-size:1.05rem; font-weight:600; color:#ff8d60; margin-bottom:0.4rem;'>{title}</div>
            <div style='font-size:0.9rem; line-height:1.45; color:#d3dde7;'>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_chroma_collection():
    """Load the ChromaDB collection. Cached to avoid reloading."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception:
            # Create collection if it doesn't exist (first run on Streamlit Cloud)
            collection = client.create_collection(name=COLLECTION_NAME)
            logger.info(f"Created new ChromaDB collection: {COLLECTION_NAME}")
        return collection
    except Exception as e:
        logger.error(f"Error loading or creating collection: {e}")
        return None


def process_uploaded_file(uploaded_file, collection):
    """
    Process an uploaded file and add it to the knowledge base.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        collection: ChromaDB collection to add documents to.
    """
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_file.name).suffix
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Extract text
        text = load_text_from_file(tmp_path)

        if not text.strip():
            st.warning(f"No text extracted from {uploaded_file.name}")
            os.unlink(tmp_path)
            return 0

        # Split into chunks (tuned for fewer, larger chunks to reduce API calls)
        chunks = split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        # Generate embeddings (parallelized)
        with st.spinner(
            f"Generating embeddings for {len(chunks)} chunks (workers={EMBED_WORKERS})..."
        ):
            embeddings = gemini_embed(chunks, max_workers=EMBED_WORKERS)

        # Prepare data
        filename = uploaded_file.name
        ids = [f"{filename}__{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": filename,
                "chunk_id": f"{filename}__{i}",
                "text_preview": chunk[:300],
            }
            for i, chunk in enumerate(chunks)
        ]

        # Add to collection
        collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks
        )

        # Cleanup
        os.unlink(tmp_path)

        return len(chunks)

    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        st.error(f"Error processing file: {e}")
        return 0


def query_knowledge_base(question: str, collection) -> dict:
    """
    Query the knowledge base and generate an answer.

    Args:
        question: User's question.
        collection: ChromaDB collection to query.

    Returns:
        Dictionary with 'answer' and 'sources' keys.
    """
    try:
        # Generate query embedding
        with st.spinner("Searching knowledge base..."):
            query_embedding = gemini_embed([question])[0]

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        # Extract results
        if not results["documents"] or not results["documents"][0]:
            return {
                "answer": "I don't know based on the provided documents.",
                "sources": [],
            }

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Debug logging
        logger.info(f"Retrieved {len(documents)} documents with distances: {distances}")

        # Filter by similarity threshold
        # ChromaDB with cosine metric returns squared L2 distances (0 = identical, larger = less similar)
        # Lower distance = higher similarity, so we filter by distance < threshold
        filtered_sources = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # For ChromaDB cosine space, distance range is typically 0-2
            # Lower distance means more similar
            # Convert to similarity percentage for display (inverse of distance)
            similarity_display = max(0, 1 - (dist / 2))  # Normalize to 0-1 range

            # Apply both distance threshold and minimum similarity threshold
            if dist <= SIMILARITY_THRESHOLD and similarity_display >= MIN_SIMILARITY:
                filtered_sources.append(
                    {
                        "text": doc,
                        "metadata": meta,
                        "similarity": similarity_display,
                        "distance": dist,
                    }
                )
                logger.debug(
                    f"Accepted: {meta['chunk_id']}, dist={dist:.4f}, sim={similarity_display:.2%}"
                )
            else:
                logger.debug(
                    f"Filtered out: {meta['chunk_id']}, dist={dist:.4f}, sim={similarity_display:.2%}"
                )

        if not filtered_sources:
            logger.warning(
                f"No relevant documents found above similarity threshold (min={MIN_SIMILARITY})"
            )
            return {
                "answer": f"I couldn't find relevant information in the documents. All retrieved chunks had similarity below {MIN_SIMILARITY:.0%}.",
                "sources": [],
            }

        # Limit to top 3 sources for the prompt
        top_sources = filtered_sources[:MAX_SOURCES_IN_PROMPT]

        # Build prompt
        system_instruction = """You are a helpful assistant that answers questions based on the provided source documents.

IMPORTANT RULES:
1. Answer based ONLY on information from the SOURCE documents below
2. DO NOT include any citations or source references in your answer
3. If the exact topic isn't mentioned in the sources, clearly state: "The provided documents do not contain information about [topic]."
4. Do NOT make up, infer, or use external knowledge not present in the sources
5. Be specific and accurate - quote or paraphrase from the sources
6. If sources mention related but not identical topics, acknowledge that explicitly
7. Provide a clean answer without any bracketed references like [source_filename] or [chunk_id]"""

        sources_text = "\n\n".join(
            [
                f"SOURCE {i+1} [{src['metadata']['chunk_id']}]:\n{src['text']}"
                for i, src in enumerate(top_sources)
            ]
        )

        user_prompt = f"""Based on these sources:

{sources_text}

Question: {question}

Answer:"""

        # Debug logging
        logger.info(f"Number of sources in prompt: {len(top_sources)}")
        logger.info(f"Question: {question}")
        logger.info(f"First source preview: {top_sources[0]['text'][:100]}...")

        # DEBUG: Dump prompt to file for inspection
        if DEBUG_DUMP_PROMPTS:
            try:
                full_prompt = f"{system_instruction}\n\n{user_prompt}"
                with open("last_prompt.txt", "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write("PROMPT SENT TO GEMINI API\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(full_prompt)
                    f.write("\n\n" + "=" * 80 + "\n")
                    f.write(f"Question: {question}\n")
                    f.write(f"Sources used: {len(top_sources)}\n")
                    f.write(f"Distances: {[s['distance'] for s in top_sources]}\n")
                logger.info("✓ Prompt dumped to last_prompt.txt")
            except Exception as e:
                logger.error(f"Failed to dump prompt: {e}")

        # DEBUG: Bypass LLM mode (return concatenated sources directly)
        if DEBUG_BYPASS_LLM:
            logger.warning(
                "DEBUG_BYPASS_LLM=True: Returning sources without LLM generation"
            )
            answer = "[DEBUG MODE: LLM BYPASSED]\n\n" + "\n\n---\n\n".join(
                [
                    f"SOURCE {i+1} [{s['metadata']['chunk_id']}]:\n{s['text']}"
                    for i, s in enumerate(top_sources)
                ]
            )
        else:
            # Generate answer
            with st.spinner("Generating answer..."):
                answer = gemini_chat(system_instruction, user_prompt, temperature=0.0)

        return {
            "answer": answer,
            "sources": filtered_sources[:3],  # Return top 3 for display
        }

    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        return {"answer": f"Error processing your question: {str(e)}", "sources": []}


def main():
    """Main Streamlit application."""
    # Header section in a styled card
    render_card(
        "🤖 Knowledge Base Agent",
        "Ask questions about your uploaded documents. Semantic retrieval + Gemini powered answers with transparent source attribution.",
        accent=True,
    )

    # Disclaimer
    st.markdown(
        """
        <div class='kb-card' style='margin-top:1rem;'>
            <strong style='color:#ff8d60;'>Disclaimer:</strong> Answers are generated from your documents. Always verify using the cited source snippets. The model will refuse to fabricate unsupported content.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load collection
    collection = load_chroma_collection()

    if collection is None:
        st.error(
            "❌ Knowledge base not found. Please run `python ingest.py` first to index your documents."
        )
        st.stop()

    # Get collection stats
    try:
        collection_count = collection.count()
        st.success(f"✓ Knowledge base loaded ({collection_count} chunks indexed)")
    except Exception as e:
        st.error(f"Error accessing knowledge base: {e}")
        st.stop()

    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Documents")
        st.markdown("Upload PDF, DOCX, or TXT files to add to the knowledge base.")

        # Clear database option
        st.markdown("---")
        st.subheader("🗑️ Database Management")

        if st.button(
            "🗑️ Clear All Documents", type="secondary", use_container_width=True
        ):
            try:
                # Get ChromaDB client
                client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                # Delete existing collection
                client.delete_collection(name=COLLECTION_NAME)
                # Create fresh collection
                client.create_collection(
                    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
                )
                st.success("✓ Database cleared! Upload new documents below.")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing database: {e}")

        st.caption("⚠️ This will remove all indexed documents")

        st.markdown("---")
        st.subheader("📤 Upload New Documents")

        uploaded_files = st.file_uploader(
            "Choose files", type=["pdf", "docx", "txt"], accept_multiple_files=True
        )

        if st.button(
            "🔄 Index Uploaded Files",
            disabled=not uploaded_files,
            use_container_width=True,
        ):
            if uploaded_files:
                total_chunks = 0
                with st.spinner("Processing uploaded files..."):
                    for uploaded_file in uploaded_files:
                        st.write(f"Processing {uploaded_file.name}...")
                        chunks_added = process_uploaded_file(uploaded_file, collection)
                        total_chunks += chunks_added

                if total_chunks > 0:
                    st.success(
                        f"✓ Added {total_chunks} chunks from {len(uploaded_files)} file(s)"
                    )
                    # Clear cache to reload collection stats
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.warning("No chunks were added. Please check the files.")

        st.caption("💡 Tip: Clear database first to query only your uploaded documents")

        # Show current documents in database
        st.markdown("---")
        st.subheader("📚 Current Documents")
        try:
            all_items = collection.get(include=["metadatas"])
            if all_items and all_items["ids"]:
                # Get unique sources
                sources = set()
                for metadata in all_items["metadatas"]:
                    sources.add(metadata.get("source", "Unknown"))

                st.write(f"**Total chunks**: {len(all_items['ids'])}")
                st.write(f"**Documents** ({len(sources)}):")
                for source in sorted(sources):
                    # Count chunks per source
                    chunk_count = sum(
                        1 for m in all_items["metadatas"] if m.get("source") == source
                    )
                    st.caption(f"  • {source} ({chunk_count} chunks)")
            else:
                st.caption("No documents indexed")
        except Exception as e:
            st.caption(f"Could not load document list: {e}")

    # Main chat interface
    st.markdown("<hr class='section-divider' />", unsafe_allow_html=True)
    render_card(
        "💬 Ask a Question",
        "Enter a natural language question. Retrieval will gather the most relevant chunks; the model then crafts an answer citing them.",
    )

    # Initialize session state for answers
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None

    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of the documents?",
        key="question_input",
    )

    # Ask button and clear button
    col1, col2 = st.columns([3, 1])
    with col1:
        ask_clicked = st.button("🚀 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.last_question = ""
            st.session_state.last_answer = None
            st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            # Query the knowledge base (always fresh, no caching)
            with st.spinner("🔍 Searching and generating answer..."):
                result = query_knowledge_base(question, collection)
            # Store in session state
            st.session_state.last_question = question
            st.session_state.last_answer = result

    # Display results if available
    if st.session_state.last_answer:
        result = st.session_state.last_answer

        # Answer Card
        render_card(
            "📝 Answer",
            f"<div style='white-space:pre-wrap;'>{result['answer']}</div>",
        )

        # Sources Section
        if result["sources"]:
            similarities = [s["similarity"] for s in result["sources"]]
            distances = [s["distance"] for s in result["sources"]]
            st.markdown("<hr class='section-divider' />", unsafe_allow_html=True)
            sim_html = (
                "These chunks were retrieved as the most relevant context."
                + f"<div class='sim-metrics'>Similarity: {[f'{s:.1%}' for s in similarities]} | Distances: {[f'{d:.3f}' for d in distances]}</div>"
            )
            render_card("📚 Source Snippets", sim_html)

            for i, source in enumerate(result["sources"]):
                with st.expander(
                    f"Source {i+1}: {source['metadata']['source']} (Similarity: {source['similarity']:.2%})"
                ):
                    st.markdown(
                        f"<div style='font-size:0.8rem;color:#95a9bc;'>Chunk ID: <code>{source['metadata']['chunk_id']}</code></div>",
                        unsafe_allow_html=True,
                    )
                    preview = (
                        source["text"][:300] + "..."
                        if len(source["text"]) > 300
                        else source["text"]
                    )
                    st.markdown(
                        f"<div style='font-size:0.82rem; line-height:1.4; color:#d3dde7; white-space:pre-wrap;'>{preview}</div>",
                        unsafe_allow_html=True,
                    )
                    if len(source["text"]) > 300:
                        if st.button("Show full text", key=f"show_full_{i}"):
                            st.markdown(
                                f"<div style='font-size:0.78rem; line-height:1.4; color:#c9d6e2; white-space:pre-wrap;'>{source['text']}</div>",
                                unsafe_allow_html=True,
                            )

    # Footer
    st.markdown(
        "<div class='footer-note'>Built with Streamlit · Gemini API · ChromaDB</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
