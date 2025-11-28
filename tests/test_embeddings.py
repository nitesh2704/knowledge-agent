"""
Test script to verify embeddings are generated correctly.
Tests for identical embeddings bug, validates types, and measures similarity.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import gemini_embed


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def test_embeddings():
    """Test embedding generation with different texts."""
    print("=" * 80)
    print("EMBEDDINGS DIAGNOSTIC TEST")
    print("=" * 80)

    # Test 1: Three different texts
    test_texts = [
        "Artificial Intelligence is the simulation of human intelligence by machines.",
        "The weather today is sunny with clear blue skies.",
        "Python is a popular programming language for data science.",
    ]

    print("\n[TEST 1] Generating embeddings for 3 different texts...")
    print("-" * 80)

    try:
        embeddings = gemini_embed(test_texts)
        print(f"✓ Successfully generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"✗ ERROR: Failed to generate embeddings: {e}")
        return False

    # Test 2: Validate shape and types
    print("\n[TEST 2] Validating embedding properties...")
    print("-" * 80)

    for i, emb in enumerate(embeddings):
        print(f"\nEmbedding {i+1}:")
        print(f"  - Type: {type(emb)}")
        print(f"  - Length (dimensions): {len(emb)}")
        print(f"  - First element type: {type(emb[0])}")
        print(f"  - First 5 values: {emb[:5]}")
        print(f"  - Last 5 values: {emb[-5:]}")
        print(f"  - Min value: {min(emb):.6f}")
        print(f"  - Max value: {max(emb):.6f}")
        print(f"  - Mean value: {np.mean(emb):.6f}")
        print(f"  - Std dev: {np.std(emb):.6f}")

        # Check if all values are floats
        all_floats = all(isinstance(x, (float, np.floating)) for x in emb)
        print(f"  - All values are floats: {all_floats}")

        if not all_floats:
            print(
                f"  ⚠️  WARNING: Not all values are floats! Types: {set(type(x) for x in emb[:10])}"
            )

    # Test 3: Check for identical embeddings (BUG DETECTION)
    print("\n[TEST 3] Checking for identical embeddings bug...")
    print("-" * 80)

    identical_count = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if embeddings[i] == embeddings[j]:
                identical_count += 1
                print(f"  ⚠️  WARNING: Embeddings {i+1} and {j+1} are IDENTICAL!")

    if identical_count == 0:
        print("  ✓ No identical embeddings found (expected for different texts)")
    else:
        print(
            f"  ✗ CRITICAL BUG: Found {identical_count} pairs of identical embeddings!"
        )

    # Test 4: Cosine similarity matrix
    print("\n[TEST 4] Computing cosine similarity matrix...")
    print("-" * 80)

    print("\nSimilarity Matrix (higher = more similar, 1.0 = identical):")
    print("        Text1   Text2   Text3")
    for i in range(len(embeddings)):
        row = f"Text{i+1}  "
        for j in range(len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            row += f"{sim:.4f}  "
        print(row)

    print("\nInterpretation:")
    print("  - Diagonal should be ~1.0 (text similar to itself)")
    print("  - Off-diagonal should be < 0.9 (different texts should differ)")

    # Calculate average off-diagonal similarity
    off_diag_sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            off_diag_sims.append(sim)

    avg_diff = np.mean(off_diag_sims)
    print(f"\n  Average similarity between different texts: {avg_diff:.4f}")

    if avg_diff > 0.95:
        print(f"  ✗ CRITICAL: Embeddings are too similar ({avg_diff:.4f})!")
        print("     This indicates all texts are getting the same embedding.")
        return False
    else:
        print(f"  ✓ Embeddings show expected variation")

    # Test 5: Test with duplicate text
    print("\n[TEST 5] Testing with duplicate text (should be nearly identical)...")
    print("-" * 80)

    duplicate_texts = [
        "Artificial Intelligence is transforming the world.",
        "Artificial Intelligence is transforming the world.",  # Exact duplicate
    ]

    try:
        dup_embeddings = gemini_embed(duplicate_texts)
        dup_sim = cosine_similarity(dup_embeddings[0], dup_embeddings[1])
        print(f"  Similarity between identical texts: {dup_sim:.6f}")

        if dup_sim > 0.999:
            print(
                f"  ✓ Identical texts produce nearly identical embeddings ({dup_sim:.6f})"
            )
        else:
            print(
                f"  ⚠️  WARNING: Identical texts have similarity {dup_sim:.6f} (expected >0.999)"
            )

    except Exception as e:
        print(f"  ✗ ERROR: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    all_same_dim = len(set(len(emb) for emb in embeddings)) == 1
    all_float = all(isinstance(emb[0], (float, np.floating)) for emb in embeddings)

    print(f"✓ Embeddings generated: {len(embeddings)}")
    print(f"✓ All same dimension: {all_same_dim}")
    print(f"✓ All values are floats: {all_float}")
    print(f"✓ No identical embeddings for different texts: {identical_count == 0}")
    print(f"✓ Reasonable variation: {avg_diff < 0.95}")

    if identical_count > 0 or avg_diff > 0.95:
        print("\n⚠️  EMBEDDINGS BUG DETECTED!")
        print("   Recommendation: Check gemini_embed() response parsing")
        return False
    else:
        print("\n✓ EMBEDDINGS TEST PASSED!")
        return True


if __name__ == "__main__":
    success = test_embeddings()
    sys.exit(0 if success else 1)
