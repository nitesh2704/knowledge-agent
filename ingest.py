"""
Document ingestion script for the Knowledge Base Agent.
Processes PDF, DOCX, and TXT files, creates embeddings, and stores them in ChromaDB.

Usage:
    python ingest.py                    # Process all files in docs/ folder
    python ingest.py --file path.pdf    # Process a single file
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict
import chromadb
from tqdm import tqdm
from utils import load_text_from_file, split_text, gemini_embed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DOCS_FOLDER = "docs"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "kb_collection"
CHUNK_SIZE = 800
OVERLAP = 100


def process_file(file_path: str) -> List[Dict]:
    """
    Process a single file: extract text, split into chunks, and prepare metadata.

    Args:
        file_path: Path to the file to process.

    Returns:
        List of dictionaries containing chunk text and metadata.
    """
    logger.info(f"Processing file: {file_path}")

    try:
        # Extract text
        text = load_text_from_file(file_path)

        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")
            return []

        # Split into chunks
        chunks = split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        logger.info(f"Split into {len(chunks)} chunks")

        # Prepare chunk data with metadata
        filename = Path(file_path).name
        chunk_data = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}__{i}"
            chunk_info = {
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": chunk_id,
                    "text_preview": chunk[:300],
                },
            }
            chunk_data.append(chunk_info)
            # Log chunk ID to verify uniqueness
            logger.debug(f"Created chunk: {chunk_id}, length: {len(chunk)}")

        # Verify all chunk IDs are unique
        chunk_ids = [c["metadata"]["chunk_id"] for c in chunk_data]
        unique_ids = set(chunk_ids)
        if len(unique_ids) != len(chunk_ids):
            logger.warning(
                f"Duplicate chunk IDs detected! Total: {len(chunk_ids)}, Unique: {len(unique_ids)}"
            )
        else:
            logger.info(f"✓ All {len(chunk_ids)} chunk IDs are unique")

        return chunk_data

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []


def ingest_documents(file_paths: List[str]) -> None:
    """
    Ingest multiple documents into the ChromaDB collection.

    Args:
        file_paths: List of file paths to ingest.
    """
    if not file_paths:
        logger.warning("No files to process")
        return

    logger.info(f"Starting ingestion of {len(file_paths)} files")

    # Process all files and collect chunks
    all_chunk_data = []
    for file_path in file_paths:
        chunk_data = process_file(file_path)
        all_chunk_data.extend(chunk_data)

    if not all_chunk_data:
        logger.warning("No chunks generated from any files")
        return

    logger.info(f"Total chunks to embed: {len(all_chunk_data)}")

    # Extract texts for embedding
    texts = [chunk["text"] for chunk in all_chunk_data]

    # Generate embeddings with progress bar
    logger.info("Generating embeddings...")
    try:
        embeddings = gemini_embed(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return

    if len(embeddings) != len(all_chunk_data):
        logger.error(
            f"Mismatch: {len(embeddings)} embeddings vs {len(all_chunk_data)} chunks"
        )
        return

    # Initialize ChromaDB
    logger.info(f"Connecting to ChromaDB at {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Get or create collection
    try:
        # Delete existing collection if it exists (for fresh ingestion)
        try:
            client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        logger.info(f"Created collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return

    # Prepare data for ChromaDB
    ids = [chunk["metadata"]["chunk_id"] for chunk in all_chunk_data]
    metadatas = [chunk["metadata"] for chunk in all_chunk_data]
    documents = texts  # Store original text for retrieval

    # Add to collection in batches
    batch_size = 100
    logger.info("Adding chunks to ChromaDB...")

    for i in tqdm(range(0, len(ids), batch_size), desc="Upserting batches"):
        batch_end = min(i + batch_size, len(ids))

        try:
            collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                documents=documents[i:batch_end],
            )
        except Exception as e:
            logger.error(f"Error adding batch {i // batch_size}: {e}")

    logger.info(f"✓ Ingestion complete!")
    logger.info(f"  Files processed: {len(file_paths)}")
    logger.info(f"  Total chunks: {len(all_chunk_data)}")
    logger.info(f"  ChromaDB path: {CHROMA_DB_PATH}")


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Knowledge Base"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single file to ingest (instead of docs/ folder)",
    )
    args = parser.parse_args()

    # Determine which files to process
    if args.file:
        # Single file mode
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {args.file}")
            return
        if file_path.suffix.lower() not in [".pdf", ".docx", ".txt"]:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return
        file_paths = [str(file_path)]
    else:
        # Folder mode
        docs_folder = Path(DOCS_FOLDER)
        if not docs_folder.exists():
            logger.info(f"Creating docs folder: {DOCS_FOLDER}")
            docs_folder.mkdir(parents=True)
            logger.warning(
                f"Please add PDF, DOCX, or TXT files to the {DOCS_FOLDER} folder and run again"
            )
            return

        # Find all supported files
        file_paths = []
        for ext in ["*.pdf", "*.docx", "*.txt"]:
            file_paths.extend([str(p) for p in docs_folder.glob(ext)])

        if not file_paths:
            logger.warning(f"No PDF, DOCX, or TXT files found in {DOCS_FOLDER}")
            return

    # Run ingestion
    ingest_documents(file_paths)


if __name__ == "__main__":
    main()
