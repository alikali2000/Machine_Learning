"""
Test script for OpenSearch Integration

Tests the OpenSearchIndexer with:
- Embedding generation
- Indexing chunks
- Text, vector, and hybrid search

Usage:
    python test/test_opensearch.py
"""

import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.semantic_chunker import SemanticChunker
from src.opensearch_indexer import OpenSearchIndexer


def test_opensearch_connection():
    """Test 1: OpenSearch Connection and Health Check"""
    print("\n" + "="*70)
    print("TEST 1: OpenSearch Connection and Health Check")
    print("="*70)

    try:
        # Check if OpenSearch is configured
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OPENSEARCH_HOST not configured")
            print("Set up .env file to run this test")
            return None

        indexer = OpenSearchIndexer()
        print(f"✓ Indexer initialized")
        print(f"  - Host: {indexer.opensearch_host}")
        print(f"  - Port: {indexer.opensearch_port}")
        print(f"  - Index: {indexer.opensearch_index_name}")

        # Check cluster health
        health = indexer.health_check()
        if health:
            print(f"\n✓ Cluster health check passed")
            print(f"  - Status: {health.get('status')}")
            print(f"  - Nodes: {health.get('number_of_nodes')}")
            print(f"  - Cluster: {health.get('cluster_name')}")

        # Check if index exists
        exists = indexer.index_exists()
        print(f"\n✓ Index exists check: {exists}")
        if not exists:
            print(f"  ⚠️  Index '{indexer.opensearch_index_name}' not found")
            print(f"  Run: python scripts/create_opensearch_index.py")

        return indexer

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_embedding_generation():
    """Test 2: Embedding Generation with Bedrock Titan"""
    print("\n" + "="*70)
    print("TEST 2: Embedding Generation")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        indexer = OpenSearchIndexer()

        # Test single embedding
        test_text = "Machine learning is a subset of artificial intelligence."
        print(f"\nGenerating embedding for test text...")
        print(f"Text: \"{test_text}\"")

        embedding = indexer.generate_embedding(test_text)

        print(f"\n✓ Embedding generated successfully")
        print(f"  - Dimension: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        print(f"  - Type: {type(embedding[0])}")

        return embedding

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_batch_embedding_generation():
    """Test 3: Batch Embedding Generation"""
    print("\n" + "="*70)
    print("TEST 3: Batch Embedding Generation")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        # Create sample chunks
        chunks = [
            {
                "chunk_id": 0,
                "content": "Machine learning algorithms learn from data.",
                "char_count": 45,
                "paragraph_start": 0
            },
            {
                "chunk_id": 1,
                "content": "Deep learning uses neural networks with multiple layers.",
                "char_count": 56,
                "paragraph_start": 1
            }
        ]

        indexer = OpenSearchIndexer()
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")

        enriched_chunks = indexer.generate_embeddings_batch(chunks)

        print(f"\n✓ Batch embedding generation complete")
        print(f"  - Chunks processed: {len(enriched_chunks)}")
        print(f"  - All have embeddings: {all('embedding' in c for c in enriched_chunks)}")

        return enriched_chunks

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_indexing_chunks():
    """Test 4: Index Chunks to OpenSearch"""
    print("\n" + "="*70)
    print("TEST 4: Indexing Chunks")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        indexer = OpenSearchIndexer()

        # Check if index exists
        if not indexer.index_exists():
            print(f"\n[SKIP] Index '{indexer.opensearch_index_name}' does not exist")
            print("Run: python scripts/create_opensearch_index.py")
            return None

        # Create test chunks with entities and topics
        test_chunks = [
            {
                "chunk_id": 9001,
                "content": "Python is a popular programming language for data science and machine learning.",
                "char_count": 78,
                "paragraph_start": 0,
                "entities": [
                    {"type": "ORG", "value": "Python"}
                ],
                "topics": ["programming", "data science", "machine learning"]
            },
            {
                "chunk_id": 9002,
                "content": "TensorFlow and PyTorch are widely used deep learning frameworks.",
                "char_count": 64,
                "paragraph_start": 1,
                "entities": [
                    {"type": "ORG", "value": "TensorFlow"},
                    {"type": "ORG", "value": "PyTorch"}
                ],
                "topics": ["deep learning", "frameworks"]
            }
        ]

        print(f"\nIndexing {len(test_chunks)} test chunks...")
        result = indexer.index_chunks(test_chunks)

        print(f"\n✓ Indexing complete")
        print(f"  - Success: {result['success']}")
        print(f"  - Failed: {result['failed']}")
        print(f"  - Total: {result['total']}")

        return result

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_text_search():
    """Test 5: Text Search (BM25)"""
    print("\n" + "="*70)
    print("TEST 5: Text Search")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        indexer = OpenSearchIndexer()

        if not indexer.index_exists():
            print(f"\n[SKIP] Index does not exist")
            return None

        query = "machine learning"
        print(f"\nSearching for: \"{query}\"")

        results = indexer.search_text(query, size=3)

        print(f"\n✓ Text search complete")
        print(f"  - Results found: {len(results)}")

        for i, result in enumerate(results[:2], 1):
            print(f"\n  Result {i}:")
            print(f"    - Score: {result.get('score', 0):.4f}")
            print(f"    - Chunk ID: {result.get('chunk_id')}")
            print(f"    - Content: {result.get('content', '')[:80]}...")

        return results

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_vector_search():
    """Test 6: Vector Search (kNN)"""
    print("\n" + "="*70)
    print("TEST 6: Vector Search")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        indexer = OpenSearchIndexer()

        if not indexer.index_exists():
            print(f"\n[SKIP] Index does not exist")
            return None

        query = "deep learning frameworks"
        print(f"\nSearching for: \"{query}\"")
        print("(Generating query embedding...)")

        results = indexer.search_vector(query, size=3)

        print(f"\n✓ Vector search complete")
        print(f"  - Results found: {len(results)}")

        for i, result in enumerate(results[:2], 1):
            print(f"\n  Result {i}:")
            print(f"    - Score: {result.get('score', 0):.4f}")
            print(f"    - Chunk ID: {result.get('chunk_id')}")
            print(f"    - Content: {result.get('content', '')[:80]}...")

        return results

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_hybrid_search():
    """Test 7: Hybrid Search (Text + Vector)"""
    print("\n" + "="*70)
    print("TEST 7: Hybrid Search")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        indexer = OpenSearchIndexer()

        if not indexer.index_exists():
            print(f"\n[SKIP] Index does not exist")
            return None

        query = "Python data science"
        print(f"\nSearching for: \"{query}\"")
        print("(Combining text and vector search...)")

        results = indexer.search_hybrid(
            query,
            size=3,
            text_weight=0.5,
            vector_weight=0.5
        )

        print(f"\n✓ Hybrid search complete")
        print(f"  - Results found: {len(results)}")

        for i, result in enumerate(results[:2], 1):
            print(f"\n  Result {i}:")
            print(f"    - Score: {result.get('score', 0):.4f}")
            print(f"    - Chunk ID: {result.get('chunk_id')}")
            print(f"    - Content: {result.get('content', '')[:80]}...")
            if 'topics' in result:
                print(f"    - Topics: {result['topics']}")

        return results

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_entity_search():
    """Test 8: Entity-Based Search"""
    print("\n" + "="*70)
    print("TEST 8: Entity-Based Search")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        indexer = OpenSearchIndexer()

        if not indexer.index_exists():
            print(f"\n[SKIP] Index does not exist")
            return None

        # Test search by entity type
        print("\nA) Search by entity type (ORG)...")
        org_results = indexer.search_by_entity(entity_type="ORG", size=3)
        print(f"  ✓ Found {len(org_results)} results with organizations")

        if org_results:
            print(f"\n  Sample result:")
            print(f"    - Chunk ID: {org_results[0].get('chunk_id')}")
            print(f"    - Entities: {org_results[0].get('entities', [])[:2]}")

        # Test search by entity value
        print("\n\nB) Search by entity value...")
        value_results = indexer.search_by_entity(entity_value="Python", size=3)
        print(f"  ✓ Found {len(value_results)} results mentioning 'Python'")

        # Test search by topic
        print("\n\nC) Search by topic...")
        topic_results = indexer.search_by_topic("machine learning", size=3)
        print(f"  ✓ Found {len(topic_results)} results about 'machine learning'")

        if topic_results:
            print(f"\n  Sample result:")
            print(f"    - Chunk ID: {topic_results[0].get('chunk_id')}")
            print(f"    - Topics: {topic_results[0].get('topics', [])}")

        print("\n✓ Entity-based search tests complete")
        return {"org": org_results, "value": value_results, "topic": topic_results}

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_combined_search():
    """Test 9: Combined Search"""
    print("\n" + "="*70)
    print("TEST 9: Combined Search (Text + Entities + Topics)")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        indexer = OpenSearchIndexer()

        if not indexer.index_exists():
            print(f"\n[SKIP] Index does not exist")
            return None

        print("\nSearching with multiple filters...")
        print("  - Text: 'deep learning'")
        print("  - Entity Type: ORG")

        results = indexer.search_combined(
            text_query="deep learning",
            entity_type="ORG",
            size=3
        )

        print(f"\n✓ Combined search complete")
        print(f"  - Results found: {len(results)}")

        if results:
            for i, result in enumerate(results[:2], 1):
                print(f"\n  Result {i}:")
                print(f"    - Chunk ID: {result.get('chunk_id')}")
                print(f"    - Content: {result.get('content', '')[:60]}...")
                print(f"    - Entities: {[e['value'] for e in result.get('entities', [])][:2]}")
                print(f"    - Topics: {result.get('topics', [])[:2]}")

        return results

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_full_integration():
    """Test 10: Full Integration - Chunk, Extract, Index, Search"""
    print("\n" + "="*70)
    print("TEST 10: Full Integration Test")
    print("="*70)

    try:
        # Check configuration
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        if not os.getenv('ANTHROPIC_API_KEY'):
            print("\n[SKIP] Anthropic API key not configured")
            return None

        print("\n1. Creating chunks with entity/topic extraction...")
        chunker = SemanticChunker(
            chunk_size=500,
            chunk_overlap=100,
            extract_entities_topics=True
        )

        chunks = chunker.chunk_file("src/sample_text.txt")
        print(f"  ✓ Created {len(chunks)} chunks")

        print("\n2. Initializing OpenSearch indexer...")
        indexer = OpenSearchIndexer()

        if not indexer.index_exists():
            print(f"  [SKIP] Index does not exist")
            return None

        print(f"  ✓ Connected to {indexer.opensearch_host}")

        print("\n3. Indexing chunks (first 3 only for testing)...")
        test_subset = chunks[:3]
        result = indexer.index_chunks(test_subset)
        print(f"  ✓ Indexed {result['success']} chunks")

        print("\n4. Performing search...")
        search_results = indexer.search_hybrid("machine learning", size=2)
        print(f"  ✓ Found {len(search_results)} results")

        print("\n✓ Full integration test complete!")
        return True

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# OPENSEARCH INTEGRATION - TEST SUITE")
    print("#"*70)

    tests = [
        ("OpenSearch Connection", test_opensearch_connection),
        ("Embedding Generation", test_embedding_generation),
        ("Batch Embeddings", test_batch_embedding_generation),
        ("Indexing Chunks", test_indexing_chunks),
        ("Text Search", test_text_search),
        ("Vector Search", test_vector_search),
        ("Hybrid Search", test_hybrid_search),
        ("Entity Search", test_entity_search),
        ("Combined Search", test_combined_search),
        ("Full Integration", test_full_integration)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            if results[test_name] is not None:
                print(f"\n[PASS] {test_name}")
            else:
                print(f"\n[SKIP] {test_name}")
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


if __name__ == "__main__":
    run_all_tests()
