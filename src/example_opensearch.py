"""
Example Usage: OpenSearch Integration with Semantic Chunking

This script demonstrates the complete workflow:
1. Chunk a document with entity/topic extraction
2. Generate embeddings using AWS Bedrock Titan
3. Index chunks to OpenSearch
4. Perform text, vector, and hybrid searches

Usage:
    python src/example_opensearch.py

Requirements:
    - .env file configured with all required credentials
    - OpenSearch index created (run scripts/create_opensearch_index.py first)
    - Sample text file available
"""

from semantic_chunker import SemanticChunker
from opensearch_indexer import OpenSearchIndexer
import json


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"{'='*70}\n")


def print_search_results(results, search_type: str):
    """Print search results in a readable format."""
    print(f"\n{search_type} Results ({len(results)} found):")
    print("-" * 70)

    for i, result in enumerate(results[:3], 1):  # Show top 3 results
        print(f"\nResult {i} (Score: {result.get('score', 0):.4f}):")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Content Preview: {result.get('content', '')[:150]}...")

        if 'entities' in result and result['entities']:
            print(f"Entities: {', '.join([f\"{e['type']}: {e['value']}\" for e in result['entities'][:3]])}")

        if 'topics' in result and result['topics']:
            print(f"Topics: {', '.join(result['topics'][:3])}")

    print("-" * 70)


def main():
    """Main demonstration function."""
    print_separator("OpenSearch Integration Demo")

    # Step 1: Create chunks with entity/topic extraction
    print_separator("Step 1: Chunking Document with Entity/Topic Extraction")

    print("Initializing SemanticChunker with entity/topic extraction...")
    chunker = SemanticChunker(
        chunk_size=800,
        chunk_overlap=150,
        extract_entities_topics=True  # Enable Claude Sonnet extraction
    )

    print("Chunking sample text file...")
    chunks = chunker.chunk_file("src/sample_text.txt")

    print(f"✓ Created {len(chunks)} semantic chunks")
    print(f"✓ Extracted entities and topics for each chunk")

    # Show sample chunk
    if chunks:
        print(f"\nSample Chunk (ID: {chunks[0]['chunk_id']}):")
        print(f"  Content length: {chunks[0]['char_count']} chars")
        print(f"  Entities: {len(chunks[0].get('entities', []))} found")
        print(f"  Topics: {chunks[0].get('topics', [])}")

    # Step 2: Initialize OpenSearch Indexer
    print_separator("Step 2: Initializing OpenSearch Indexer")

    print("Connecting to OpenSearch and Bedrock...")
    indexer = OpenSearchIndexer()

    # Check if index exists
    if not indexer.index_exists():
        print(f"⚠️  Warning: Index '{indexer.opensearch_index_name}' does not exist!")
        print("Please run: python scripts/create_opensearch_index.py")
        return

    print(f"✓ Connected to OpenSearch: {indexer.opensearch_host}")
    print(f"✓ Using index: {indexer.opensearch_index_name}")
    print(f"✓ Bedrock region: {indexer.bedrock_region}")
    print(f"✓ Embedding model: {indexer.titan_model_id}")

    # Step 3: Index chunks to OpenSearch
    print_separator("Step 3: Indexing Chunks to OpenSearch")

    print("Generating embeddings and indexing chunks...")
    print("(This may take a few moments depending on the number of chunks)\n")

    result = indexer.index_chunks(chunks)

    print(f"\n✓ Indexing complete!")
    print(f"  - Total chunks: {result['total']}")
    print(f"  - Successfully indexed: {result['success']}")
    print(f"  - Failed: {result['failed']}")

    # Save chunks locally as well
    output_file = "chunks_with_embeddings.json"
    chunker.save_chunks(chunks, output_file)
    print(f"✓ Chunks also saved to {output_file}")

    # Step 4: Perform searches
    print_separator("Step 4: Searching OpenSearch")

    # Example query
    search_query = "machine learning"
    print(f"Search Query: \"{search_query}\"\n")

    # Text search (BM25)
    print("A) Text Search (BM25)...")
    text_results = indexer.search_text(search_query, size=5)
    print_search_results(text_results, "Text Search")

    # Vector search (kNN)
    print("\n\nB) Vector Search (kNN)...")
    vector_results = indexer.search_vector(search_query, size=5)
    print_search_results(vector_results, "Vector Search")

    # Hybrid search (combined)
    print("\n\nC) Hybrid Search (Text + Vector)...")
    hybrid_results = indexer.search_hybrid(
        search_query,
        size=5,
        text_weight=0.5,
        vector_weight=0.5
    )
    print_search_results(hybrid_results, "Hybrid Search")

    # Step 5: Entity-based search
    print_separator("Step 5: Entity-Based Search")

    print("A) Search by Entity Type (Organizations)...")
    org_results = indexer.search_by_entity(entity_type="ORG", size=3)
    print_search_results(org_results, "Organization Search")

    print("\n\nB) Search by Entity Value...")
    # Try to find specific entity (adjust based on your data)
    entity_results = indexer.search_by_entity(entity_value="Python", size=3)
    print_search_results(entity_results, "Entity Value Search")

    print("\n\nC) Search by Topic...")
    topic_results = indexer.search_by_topic("machine learning", size=3)
    print_search_results(topic_results, "Topic Search")

    # Step 6: Combined search
    print_separator("Step 6: Combined Search (Text + Entities + Topics)")

    print("Searching for chunks about 'neural networks' with organization mentions...")
    combined_results = indexer.search_combined(
        text_query="neural networks",
        entity_type="ORG",
        size=3
    )
    print_search_results(combined_results, "Combined Search")

    # Summary
    print_separator("Demo Complete!")

    print("Summary:")
    print(f"  ✓ Chunked document into {len(chunks)} semantic chunks")
    print(f"  ✓ Extracted entities and topics using Claude Sonnet")
    print(f"  ✓ Generated {result['success']} embeddings using AWS Bedrock Titan")
    print(f"  ✓ Indexed to OpenSearch with full metadata")
    print(f"\n  ✓ Demonstrated 6 search types:")
    print(f"    - Text search (BM25)")
    print(f"    - Vector search (kNN)")
    print(f"    - Hybrid search (Text + Vector)")
    print(f"    - Entity search (by type/value)")
    print(f"    - Topic search")
    print(f"    - Combined search (Text + Entities + Topics)")
    print(f"\nYou can now perform any of these searches on your indexed data!")

    print_separator()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("  1. .env file is properly configured")
        print("  2. AWS credentials are set up")
        print("  3. OpenSearch index exists (run create_opensearch_index.py)")
        print("  4. Anthropic API key is configured")
