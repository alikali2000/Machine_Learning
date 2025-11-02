"""
Test script for SemanticChunker
Demonstrates different ways to test the chunker with various parameters
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.semantic_chunker import SemanticChunker
import json


def test_basic_chunking():
    """Test 1: Basic chunking with default parameters"""
    print("\n" + "="*70)
    print("TEST 1: Basic Chunking with Default Parameters")
    print("="*70)

    chunker = SemanticChunker()
    chunks = chunker.chunk_file("src/sample_text.txt")
    chunker.print_chunk_summary(chunks)

    return chunks


def test_small_chunks():
    """Test 2: Smaller chunk sizes"""
    print("\n" + "="*70)
    print("TEST 2: Smaller Chunks (500 chars, 100 overlap)")
    print("="*70)

    chunker = SemanticChunker(
        chunk_size=500,
        chunk_overlap=100,
        min_chunk_size=50
    )
    chunks = chunker.chunk_file("src/sample_text.txt")
    chunker.print_chunk_summary(chunks)

    return chunks


def test_large_chunks():
    """Test 3: Larger chunk sizes"""
    print("\n" + "="*70)
    print("TEST 3: Larger Chunks (2000 chars, 300 overlap)")
    print("="*70)

    chunker = SemanticChunker(
        chunk_size=2000,
        chunk_overlap=300,
        min_chunk_size=200
    )
    chunks = chunker.chunk_file("src/sample_text.txt")
    chunker.print_chunk_summary(chunks)

    return chunks


def test_no_overlap():
    """Test 4: No overlap between chunks"""
    print("\n" + "="*70)
    print("TEST 4: No Overlap Between Chunks")
    print("="*70)

    chunker = SemanticChunker(
        chunk_size=1000,
        chunk_overlap=0,  # No overlap
        min_chunk_size=100
    )
    chunks = chunker.chunk_file("src/sample_text.txt")
    chunker.print_chunk_summary(chunks)

    return chunks


def test_notebook_chunking():
    """Test 5: Chunking Jupyter notebook"""
    print("\n" + "="*70)
    print("TEST 5: Chunking Jupyter Notebook")
    print("="*70)

    chunker = SemanticChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_file("notebooks/Decision Trees and Random Forest Project.ipynb")
    chunker.print_chunk_summary(chunks)

    return chunks


def test_chunk_content_inspection():
    """Test 6: Inspect individual chunk contents"""
    print("\n" + "="*70)
    print("TEST 6: Detailed Chunk Content Inspection")
    print("="*70)

    chunker = SemanticChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_file("src/sample_text.txt")

    print(f"\nTotal chunks created: {len(chunks)}\n")

    # Show detailed view of each chunk
    for chunk in chunks:
        print(f"\n{'-'*70}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Character Count: {chunk['char_count']}")
        print(f"Paragraph Start: {chunk['paragraph_start']}")
        print(f"{'-'*70}")
        print(f"Content:\n{chunk['content']}\n")

    return chunks


def test_save_different_formats():
    """Test 7: Save chunks and verify JSON structure"""
    print("\n" + "="*70)
    print("TEST 7: Save and Verify JSON Output")
    print("="*70)

    chunker = SemanticChunker()
    chunks = chunker.chunk_file("src/sample_text.txt")

    # Save to file
    output_file = "test_chunks.json"
    chunker.save_chunks(chunks, output_file)

    # Read back and verify
    with open(output_file, 'r', encoding='utf-8') as f:
        loaded_chunks = json.load(f)

    print(f"\n[OK] Saved {len(chunks)} chunks to {output_file}")
    print(f"[OK] Successfully loaded {len(loaded_chunks)} chunks from file")
    print(f"[OK] Data integrity verified: {chunks == loaded_chunks}")

    return chunks


def test_entity_topic_extraction():
    """Test 8: Entity and Topic Extraction with Claude Sonnet"""
    print("\n" + "="*70)
    print("TEST 8: Entity and Topic Extraction")
    print("="*70)

    try:
        # Check if API key is available
        import os
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("\n[SKIP] No ANTHROPIC_API_KEY found. Skipping this test.")
            print("To run this test, set ANTHROPIC_API_KEY environment variable.")
            return None

        chunker = SemanticChunker(
            chunk_size=800,
            chunk_overlap=150,
            extract_entities_topics=True
        )
        chunks = chunker.chunk_file("src/sample_text.txt")

        print(f"\nTotal chunks created: {len(chunks)}")
        print(f"Entity/Topic extraction enabled: {chunker.extract_entities_topics}")

        # Display first 2 chunks with their entities and topics
        for i, chunk in enumerate(chunks[:2]):
            print(f"\n{'-'*70}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Character Count: {chunk['char_count']}")
            print(f"{'-'*70}")
            print(f"Content Preview: {chunk['content'][:150]}...")

            if 'entities' in chunk:
                print(f"\nEntities ({len(chunk['entities'])} found):")
                for entity in chunk['entities']:
                    print(f"  - {entity['type']}: {entity['value']}")

            if 'topics' in chunk:
                print(f"\nTopics ({len(chunk['topics'])} found):")
                for topic in chunk['topics']:
                    print(f"  - {topic}")

        # Save chunks with entities and topics
        output_file = "test_chunks_with_entities.json"
        chunker.save_chunks(chunks, output_file)
        print(f"\n[OK] Saved chunks with entities/topics to {output_file}")

        return chunks

    except ImportError as e:
        print(f"\n[SKIP] Required packages not installed: {e}")
        return None
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        return None


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# SEMANTIC CHUNKER - COMPREHENSIVE TEST SUITE")
    print("#"*70)

    tests = [
        ("Basic Chunking", test_basic_chunking),
        ("Small Chunks", test_small_chunks),
        ("Large Chunks", test_large_chunks),
        ("No Overlap", test_no_overlap),
        ("Notebook Chunking", test_notebook_chunking),
        ("Content Inspection", test_chunk_content_inspection),
        ("Save & Verify", test_save_different_formats),
        ("Entity/Topic Extraction", test_entity_topic_extraction)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            print(f"\n[PASS] {test_name}")
        except Exception as e:
            print(f"\n[FAIL] {test_name}: {str(e)}")
            results[test_name] = None

    # Final summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"\nTests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print("\n" + "#"*70 + "\n")


def interactive_test():
    """Interactive testing - lets you specify parameters"""
    print("\n" + "="*70)
    print("INTERACTIVE TEST MODE")
    print("="*70)

    print("\nAvailable files:")
    print("1. src/sample_text.txt")
    print("2. notebooks/Decision Trees and Random Forest Project.ipynb")
    print("3. S3 file (custom S3 URI)")

    file_choice = input("\nEnter file number (1, 2, or 3): ").strip()

    if file_choice == "1":
        filename = "src/sample_text.txt"
    elif file_choice == "2":
        filename = "notebooks/Decision Trees and Random Forest Project.ipynb"
    elif file_choice == "3":
        filename = input("Enter S3 URI (e.g., s3://bucket/path/file.txt): ").strip()
        if not filename.startswith('s3://'):
            print("Warning: S3 URI should start with s3://")
    else:
        print("Invalid choice, using sample_text.txt")
        filename = "src/sample_text.txt"

    chunk_size = input("\nEnter chunk size (default 1000): ").strip()
    chunk_size = int(chunk_size) if chunk_size else 1000

    overlap = input("Enter overlap size (default 200): ").strip()
    overlap = int(overlap) if overlap else 200

    min_size = input("Enter minimum chunk size (default 100): ").strip()
    min_size = int(min_size) if min_size else 100

    print(f"\nChunking {filename} with:")
    print(f"  - Chunk size: {chunk_size}")
    print(f"  - Overlap: {overlap}")
    print(f"  - Min size: {min_size}")

    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        min_chunk_size=min_size
    )

    chunks = chunker.chunk_file(filename)
    chunker.print_chunk_summary(chunks)

    save = input("\nSave chunks to file? (y/n): ").strip().lower()
    if save == 'y':
        output_name = input("Enter output filename (default: custom_chunks.json): ").strip()
        output_name = output_name if output_name else "custom_chunks.json"
        chunker.save_chunks(chunks, output_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        run_all_tests()
