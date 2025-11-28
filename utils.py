"""
Utility functions for text extraction, chunking, and Gemini API interactions.
This module is designed to be provider-agnostic for easy swapping of embedding/LLM providers.
"""

import os
import time
import logging
from typing import List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from dotenv import load_dotenv
import pdfplumber
from docx import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TODO: Update these model names if Google changes the API or you want to use different models
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_CHAT_MODEL = "models/gemini-2.0-flash"  # Using stable Gemini 2.0 Flash

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY not found in environment variables. Please set it in .env file."
    )

# TODO: Update these endpoints if Google changes their API structure
GEMINI_EMBED_ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_CHAT_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_CHAT_MODEL}:generateContent"


def load_text_from_file(file_path: str) -> str:
    """
    Extract text content from a file (PDF, DOCX, or TXT).

    Args:
        file_path: Path to the file to extract text from.

    Returns:
        Extracted text as a string.

    Raises:
        ValueError: If file type is unsupported.
        Exception: If file reading fails.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext == ".pdf":
            return _extract_pdf(file_path)
        elif ext == ".docx":
            return _extract_docx(file_path)
        elif ext == ".txt":
            return _extract_txt(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {ext}. Supported types: .pdf, .docx, .txt"
            )
    except Exception as e:
        logger.error(f"Error loading text from {file_path}: {e}")
        raise


def _extract_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def _extract_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    doc = Document(file_path)
    text_parts = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)
    return "\n".join(text_parts)


def _extract_txt(file_path: str) -> str:
    """Extract text from plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Target size for each chunk in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text or chunk_size <= 0:
        return []

    if overlap >= chunk_size:
        logger.warning(
            f"Overlap ({overlap}) >= chunk_size ({chunk_size}). Setting overlap to chunk_size // 2."
        )
        overlap = chunk_size // 2

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start position forward by (chunk_size - overlap)
        start += chunk_size - overlap

        # If we're at the end and there's remaining text, ensure we capture it
        if start < text_length and end >= text_length:
            break

    return chunks


def gemini_embed(
    texts: List[str],
    max_workers: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[List[float]]:
    """Generate embeddings for a list of texts using Gemini API.

    Optimizations:
    - Optional parallel requests (max_workers > 1) for throughput
    - Retry logic with exponential backoff per text
    - Order preservation (results align with input order)

    Args:
        texts: List of text strings to embed.
        max_workers: Concurrency level (1 = sequential). Keep modest (e.g., 4) to avoid rate limits.
        progress_callback: Optional function called with (completed, total).

    Returns:
        List of embedding vectors.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

    def _embed_single(idx: int, text: str) -> Tuple[int, List[float]]:
        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info(f"Embedding text {idx + 1}/{len(texts)}")

        payload = {
            "model": GEMINI_EMBEDDING_MODEL,
            "content": {"parts": [{"text": text}]},
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{GEMINI_EMBED_ENDPOINT_BASE}/{GEMINI_EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding", {}).get("values", [])
                    if embedding:
                        try:
                            embedding_floats = list(map(float, embedding))
                            return idx, embedding_floats
                        except (ValueError, TypeError) as e:
                            logger.error(
                                f"Embedding conversion error idx={idx}: {e}. Raw type: {type(embedding[0]) if embedding else 'empty'}"
                            )
                            raise ValueError(f"Invalid embedding format: {e}")
                    return idx, []
                elif response.status_code == 429:
                    wait_time = (2**attempt) * 2
                    logger.warning(
                        f"Rate limit (429) on idx={idx}. Waiting {wait_time}s (attempt {attempt+1}/{max_retries})."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"API error {response.status_code} for idx={idx}: {response.text}"
                    )
                    if attempt == max_retries - 1:
                        raise Exception(
                            f"Failed embedding idx={idx} after {max_retries} attempts: {response.text}"
                        )
                    time.sleep(2**attempt)
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout idx={idx} attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)
            except Exception as e:
                logger.error(f"Error embedding idx={idx}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)
        return idx, []

    if max_workers <= 1:
        embeddings: List[List[float]] = []
        for i, t in enumerate(texts):
            _, emb = _embed_single(i, t)
            embeddings.append(emb)
            if progress_callback:
                progress_callback(i + 1, len(texts))
        return embeddings

    # Parallel execution
    embeddings_ordered: List[Optional[List[float]]] = [None] * len(texts)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_embed_single, i, t): i for i, t in enumerate(texts)}
        for future in as_completed(futures):
            idx, emb = future.result()
            embeddings_ordered[idx] = emb
            completed += 1
            if progress_callback:
                progress_callback(completed, len(texts))
    # Replace any None with empty list to avoid issues
    return [emb if emb is not None else [] for emb in embeddings_ordered]


def gemini_chat(
    system_instruction: str, user_prompt: str, temperature: float = 0.0
) -> str:
    """
    Generate a chat completion using Gemini API.
    Implements retry logic with exponential backoff.

    Args:
        system_instruction: System-level instruction for the model.
        user_prompt: The user's prompt/question.
        temperature: Sampling temperature (0.0 for deterministic).

    Returns:
        The model's text response.

    Example API Response Shape:
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "The actual response text here..."
                            }
                        ]
                    }
                }
            ]
        }
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

    # Build the prompt combining system instruction and user prompt
    full_prompt = f"{system_instruction}\n\n{user_prompt}"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 2048,
        },
    }

    # Retry logic with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{GEMINI_CHAT_ENDPOINT}?key={GEMINI_API_KEY}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                # Extract text from the response
                try:
                    text_response = data["candidates"][0]["content"]["parts"][0]["text"]
                    return text_response
                except (KeyError, IndexError) as e:
                    logger.error(f"Unexpected response structure: {data}")
                    raise Exception(f"Failed to parse response: {e}")
            elif response.status_code == 429:  # Rate limit
                wait_time = (2**attempt) * 2
                logger.warning(
                    f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}"
                )
                logger.debug(f"Rate limit response: {response.text}")
                time.sleep(wait_time)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed to get chat response after {max_retries} attempts: {response.text}"
                    )
                time.sleep(2**attempt)
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
        except Exception as e:
            logger.error(f"Error during chat completion: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)

    raise Exception("Failed to get chat response after all retries")
