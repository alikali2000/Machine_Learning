"""
Create OpenSearch Index for Semantic Chunks

This script creates an OpenSearch index with the proper schema for:
- Text content (searchable with BM25)
- Vector embeddings (kNN search)
- Entity metadata (nested objects)
- Topic metadata (keyword array)

Usage:
    python scripts/create_opensearch_index.py

Requirements:
    - OpenSearch cluster must be running and accessible
    - AWS credentials configured for OpenSearch authentication
    - Environment variables set in .env file
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.opensearch_indexer import OpenSearchIndexer
from opensearchpy import OpenSearch


def create_index_with_mappings(indexer: OpenSearchIndexer, index_name: str):
    """
    Create OpenSearch index with proper mappings for semantic chunks.

    Args:
        indexer: OpenSearchIndexer instance
        index_name: Name of the index to create
    """
    client = indexer._get_opensearch_client()

    # Define index settings and mappings
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": 2,
                "number_of_replicas": 1,
                "knn": True,  # Enable k-NN plugin
                "knn.algo_param.ef_search": 512
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {
                    "type": "integer"
                },
                "content": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "char_count": {
                    "type": "integer"
                },
                "paragraph_start": {
                    "type": "integer"
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1536,  # Titan text embedding dimension
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16
                        }
                    }
                },
                "entities": {
                    "type": "nested",
                    "properties": {
                        "type": {
                            "type": "keyword"
                        },
                        "value": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        }
                    }
                },
                "topics": {
                    "type": "keyword"
                }
            }
        }
    }

    try:
        # Check if index already exists
        if client.indices.exists(index=index_name):
            print(f"⚠️  Index '{index_name}' already exists.")
            response = input("Do you want to delete and recreate it? (yes/no): ")

            if response.lower() in ['yes', 'y']:
                print(f"Deleting existing index '{index_name}'...")
                client.indices.delete(index=index_name)
                print(f"✓ Index '{index_name}' deleted.")
            else:
                print("Operation cancelled.")
                return False

        # Create the index
        print(f"\nCreating index '{index_name}' with mappings...")
        response = client.indices.create(index=index_name, body=index_body)

        if response.get('acknowledged'):
            print(f"✓ Index '{index_name}' created successfully!")
            print(f"\nIndex configuration:")
            print(f"  - Text search: Enabled (BM25)")
            print(f"  - Vector search: Enabled (kNN with cosine similarity)")
            print(f"  - Embedding dimension: 1536 (Titan)")
            print(f"  - Entity metadata: Nested objects")
            print(f"  - Topic metadata: Keywords")
            return True
        else:
            print(f"✗ Failed to create index '{index_name}'")
            return False

    except Exception as e:
        print(f"✗ Error creating index: {e}")
        return False


def verify_index(indexer: OpenSearchIndexer, index_name: str):
    """
    Verify that the index was created correctly.

    Args:
        indexer: OpenSearchIndexer instance
        index_name: Name of the index to verify
    """
    try:
        client = indexer._get_opensearch_client()

        # Get index settings
        settings = client.indices.get_settings(index=index_name)
        print(f"\n✓ Index settings retrieved successfully")

        # Get index mappings
        mappings = client.indices.get_mapping(index=index_name)
        print(f"✓ Index mappings retrieved successfully")

        # Count documents (should be 0 for new index)
        count = client.count(index=index_name)
        doc_count = count.get('count', 0)
        print(f"✓ Document count: {doc_count}")

        return True

    except Exception as e:
        print(f"✗ Error verifying index: {e}")
        return False


def main():
    """Main function to create OpenSearch index."""
    print("="*70)
    print("OpenSearch Index Creation Script")
    print("="*70)

    try:
        # Initialize indexer
        print("\nInitializing OpenSearch connection...")
        indexer = OpenSearchIndexer()
        print(f"✓ Connected to OpenSearch at {indexer.opensearch_host}")

        # Check cluster health
        print("\nChecking cluster health...")
        health = indexer.health_check()
        if health:
            status = health.get('status', 'unknown')
            print(f"✓ Cluster status: {status}")
            print(f"  - Cluster name: {health.get('cluster_name')}")
            print(f"  - Number of nodes: {health.get('number_of_nodes')}")
        else:
            print("⚠️  Could not retrieve cluster health")

        # Create index
        index_name = indexer.opensearch_index_name
        success = create_index_with_mappings(indexer, index_name)

        if success:
            # Verify index
            print("\nVerifying index...")
            verify_index(indexer, index_name)

            print("\n" + "="*70)
            print("✓ Index creation completed successfully!")
            print("="*70)
            print(f"\nYou can now index chunks using:")
            print(f"  from src.opensearch_indexer import OpenSearchIndexer")
            print(f"  indexer = OpenSearchIndexer()")
            print(f"  indexer.index_chunks(chunks)")
        else:
            print("\n" + "="*70)
            print("✗ Index creation failed")
            print("="*70)
            sys.exit(1)

    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nPlease ensure:")
        print("  1. .env file is configured with OPENSEARCH_HOST")
        print("  2. AWS credentials are set up")
        print("  3. OpenSearch cluster is accessible")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
