"""
OpenSearch Indexer Module

Handles indexing of semantic chunks to AWS OpenSearch with:
- Vector embeddings using AWS Bedrock Titan
- Text, vector, and hybrid search capabilities
- Entity and topic metadata storage
"""

import os
import json
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False

# Load environment variables
load_dotenv()


class OpenSearchIndexer:
    """
    Indexes semantic chunks to AWS OpenSearch with vector embeddings.

    Supports:
    - Text search (BM25)
    - Vector similarity search (kNN)
    - Hybrid search (combined text + vector)
    """

    def __init__(
        self,
        opensearch_host: Optional[str] = None,
        opensearch_port: Optional[int] = None,
        opensearch_index_name: Optional[str] = None,
        opensearch_use_ssl: Optional[bool] = None,
        bedrock_region: Optional[str] = None,
        titan_model_id: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
        """
        Initialize OpenSearch Indexer.

        Args:
            opensearch_host: OpenSearch endpoint (without https://)
            opensearch_port: Port number (default: 443 for AWS)
            opensearch_index_name: Index name (default: semantic_chunks)
            opensearch_use_ssl: Use SSL connection (default: True)
            bedrock_region: AWS region for Bedrock (default: us-east-1)
            titan_model_id: Titan embedding model ID
            aws_access_key_id: AWS access key (optional, uses env/credentials file)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional)
        """
        # Check required dependencies
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required. Install with: pip install boto3")

        if not OPENSEARCH_AVAILABLE:
            raise ImportError(
                "opensearch-py is required. Install with: pip install opensearch-py"
            )

        # OpenSearch configuration with fallback to environment variables
        self.opensearch_host = opensearch_host or os.getenv('OPENSEARCH_HOST')
        self.opensearch_port = opensearch_port or int(os.getenv('OPENSEARCH_PORT', '443'))
        self.opensearch_index_name = opensearch_index_name or os.getenv('OPENSEARCH_INDEX_NAME', 'semantic_chunks')

        use_ssl_env = os.getenv('OPENSEARCH_USE_SSL', 'true').lower() == 'true'
        self.opensearch_use_ssl = opensearch_use_ssl if opensearch_use_ssl is not None else use_ssl_env

        # AWS Bedrock configuration
        self.bedrock_region = bedrock_region or os.getenv('BEDROCK_REGION', 'us-east-1')
        self.titan_model_id = titan_model_id or os.getenv('TITAN_MODEL_ID', 'amazon.titan-embed-text-v1')

        # AWS credentials
        self.aws_access_key_id = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_session_token = aws_session_token or os.getenv('AWS_SESSION_TOKEN')

        # Validate required configuration
        if not self.opensearch_host:
            raise ValueError(
                "OpenSearch host is required. Set OPENSEARCH_HOST environment variable "
                "or pass opensearch_host parameter."
            )

        # Client caches
        self._opensearch_client = None
        self._bedrock_client = None

    def _get_bedrock_client(self):
        """
        Get or create Bedrock runtime client for embeddings.

        Returns:
            boto3.client: Configured Bedrock runtime client
        """
        if self._bedrock_client is not None:
            return self._bedrock_client

        # Build kwargs for boto3 client
        kwargs = {
            'service_name': 'bedrock-runtime',
            'region_name': self.bedrock_region
        }

        # Use explicit credentials if provided
        if self.aws_access_key_id and self.aws_secret_access_key:
            kwargs['aws_access_key_id'] = self.aws_access_key_id
            kwargs['aws_secret_access_key'] = self.aws_secret_access_key
            if self.aws_session_token:
                kwargs['aws_session_token'] = self.aws_session_token

        self._bedrock_client = boto3.client(**kwargs)
        return self._bedrock_client

    def _get_opensearch_client(self):
        """
        Get or create OpenSearch client with AWS authentication.

        Returns:
            OpenSearch: Configured OpenSearch client
        """
        if self._opensearch_client is not None:
            return self._opensearch_client

        # Get AWS credentials for signing
        if self.aws_access_key_id and self.aws_secret_access_key:
            # Use explicit credentials
            credentials = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token
            ).get_credentials()
        else:
            # Use default credential chain
            credentials = boto3.Session().get_credentials()

        # Create AWS V4 signer for authentication
        auth = AWSV4SignerAuth(credentials, self.bedrock_region, 'es')

        # Create OpenSearch client
        self._opensearch_client = OpenSearch(
            hosts=[{'host': self.opensearch_host, 'port': self.opensearch_port}],
            http_auth=auth,
            use_ssl=self.opensearch_use_ssl,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )

        return self._opensearch_client

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using AWS Bedrock Titan.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            bedrock = self._get_bedrock_client()

            # Prepare request body for Titan
            body = json.dumps({"inputText": text})

            # Invoke Titan model
            response = bedrock.invoke_model(
                modelId=self.titan_model_id,
                contentType='application/json',
                accept='application/json',
                body=body
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')

            if not embedding:
                raise ValueError("No embedding returned from Bedrock Titan")

            return embedding

        except ClientError as e:
            print(f"Error generating embedding: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error generating embedding: {e}")
            raise

    def generate_embeddings_batch(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of chunk dictionaries with 'content' field

        Returns:
            List of chunks with added 'embedding' field
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")

        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = self.generate_embedding(chunk['content'])
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embedding
                enriched_chunks.append(chunk_with_embedding)

                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{len(chunks)} embeddings...")

            except Exception as e:
                print(f"Error generating embedding for chunk {chunk.get('chunk_id', i)}: {e}")
                # Continue with other chunks
                continue

        print(f"Successfully generated {len(enriched_chunks)} embeddings")
        return enriched_chunks

    def index_chunk(self, chunk: Dict, chunk_embedding: Optional[List[float]] = None) -> bool:
        """
        Index a single chunk to OpenSearch.

        Args:
            chunk: Chunk dictionary with content and metadata
            chunk_embedding: Pre-generated embedding (if None, will generate)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = self._get_opensearch_client()

            # Generate embedding if not provided
            if chunk_embedding is None:
                chunk_embedding = self.generate_embedding(chunk['content'])

            # Prepare document for indexing
            document = {
                'chunk_id': chunk.get('chunk_id'),
                'content': chunk.get('content'),
                'char_count': chunk.get('char_count'),
                'paragraph_start': chunk.get('paragraph_start'),
                'embedding': chunk_embedding
            }

            # Add entities if present
            if 'entities' in chunk:
                document['entities'] = chunk['entities']

            # Add topics if present
            if 'topics' in chunk:
                document['topics'] = chunk['topics']

            # Index document
            response = client.index(
                index=self.opensearch_index_name,
                body=document,
                id=str(chunk.get('chunk_id', ''))
            )

            return response.get('result') in ['created', 'updated']

        except Exception as e:
            print(f"Error indexing chunk {chunk.get('chunk_id')}: {e}")
            return False

    def index_chunks(self, chunks: List[Dict]) -> Dict[str, int]:
        """
        Index multiple chunks to OpenSearch.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Dictionary with success/failure counts
        """
        print(f"\nIndexing {len(chunks)} chunks to OpenSearch...")

        # Generate embeddings first
        chunks_with_embeddings = self.generate_embeddings_batch(chunks)

        # Index each chunk
        success_count = 0
        failure_count = 0

        for chunk in chunks_with_embeddings:
            embedding = chunk.pop('embedding')  # Remove embedding from chunk dict
            if self.index_chunk(chunk, embedding):
                success_count += 1
            else:
                failure_count += 1

            if (success_count + failure_count) % 10 == 0:
                print(f"Indexed {success_count + failure_count}/{len(chunks_with_embeddings)} chunks...")

        print(f"\nIndexing complete:")
        print(f"  ✓ Successfully indexed: {success_count}")
        print(f"  ✗ Failed: {failure_count}")

        return {
            'success': success_count,
            'failed': failure_count,
            'total': len(chunks)
        }

    def search_text(self, query: str, size: int = 10) -> List[Dict]:
        """
        Perform text-based search using BM25.

        Args:
            query: Search query text
            size: Number of results to return

        Returns:
            List of matching documents with scores
        """
        try:
            client = self._get_opensearch_client()

            search_body = {
                'size': size,
                'query': {
                    'match': {
                        'content': {
                            'query': query,
                            'fuzziness': 'AUTO'
                        }
                    }
                }
            }

            response = client.search(
                index=self.opensearch_index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                results.append(result)

            return results

        except Exception as e:
            print(f"Error performing text search: {e}")
            return []

    def search_vector(self, query: str, size: int = 10) -> List[Dict]:
        """
        Perform vector similarity search using kNN.

        Args:
            query: Search query text (will be embedded)
            size: Number of results to return

        Returns:
            List of similar documents with scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)

            client = self._get_opensearch_client()

            search_body = {
                'size': size,
                'query': {
                    'knn': {
                        'embedding': {
                            'vector': query_embedding,
                            'k': size
                        }
                    }
                }
            }

            response = client.search(
                index=self.opensearch_index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                # Remove embedding from result for cleaner output
                result.pop('embedding', None)
                results.append(result)

            return results

        except Exception as e:
            print(f"Error performing vector search: {e}")
            return []

    def search_hybrid(
        self,
        query: str,
        size: int = 10,
        text_weight: float = 0.5,
        vector_weight: float = 0.5
    ) -> List[Dict]:
        """
        Perform hybrid search combining text and vector search.

        Args:
            query: Search query text
            size: Number of results to return
            text_weight: Weight for text search (default: 0.5)
            vector_weight: Weight for vector search (default: 0.5)

        Returns:
            List of documents ranked by combined score
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)

            client = self._get_opensearch_client()

            search_body = {
                'size': size,
                'query': {
                    'bool': {
                        'should': [
                            {
                                'match': {
                                    'content': {
                                        'query': query,
                                        'boost': text_weight
                                    }
                                }
                            },
                            {
                                'knn': {
                                    'embedding': {
                                        'vector': query_embedding,
                                        'k': size,
                                        'boost': vector_weight
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            response = client.search(
                index=self.opensearch_index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                # Remove embedding from result
                result.pop('embedding', None)
                results.append(result)

            return results

        except Exception as e:
            print(f"Error performing hybrid search: {e}")
            return []

    def search_by_entity(
        self,
        entity_type: Optional[str] = None,
        entity_value: Optional[str] = None,
        size: int = 10
    ) -> List[Dict]:
        """
        Search for chunks containing specific entities.

        Args:
            entity_type: Type of entity (PERSON, ORG, LOC, DATE). If None, searches all types.
            entity_value: Entity value to search for. Supports fuzzy matching.
            size: Number of results to return

        Returns:
            List of matching documents with scores

        Examples:
            # Find all chunks mentioning any organization
            results = indexer.search_by_entity(entity_type="ORG")

            # Find chunks mentioning a specific person
            results = indexer.search_by_entity(entity_type="PERSON", entity_value="John Smith")

            # Find chunks with any entity containing "OpenAI"
            results = indexer.search_by_entity(entity_value="OpenAI")
        """
        try:
            client = self._get_opensearch_client()

            # Build nested query for entities
            must_conditions = []

            if entity_type:
                must_conditions.append({
                    "match": {
                        "entities.type": entity_type
                    }
                })

            if entity_value:
                must_conditions.append({
                    "match": {
                        "entities.value": {
                            "query": entity_value,
                            "fuzziness": "AUTO"
                        }
                    }
                })

            # If no conditions specified, match all documents with entities
            if not must_conditions:
                must_conditions.append({
                    "exists": {
                        "field": "entities"
                    }
                })

            search_body = {
                'size': size,
                'query': {
                    'nested': {
                        'path': 'entities',
                        'query': {
                            'bool': {
                                'must': must_conditions
                            }
                        }
                    }
                }
            }

            response = client.search(
                index=self.opensearch_index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                result.pop('embedding', None)
                results.append(result)

            return results

        except Exception as e:
            print(f"Error performing entity search: {e}")
            return []

    def search_by_topic(self, topic: str, size: int = 10) -> List[Dict]:
        """
        Search for chunks containing a specific topic.

        Args:
            topic: Topic to search for (supports partial matching)
            size: Number of results to return

        Returns:
            List of matching documents with scores

        Example:
            results = indexer.search_by_topic("machine learning")
        """
        try:
            client = self._get_opensearch_client()

            search_body = {
                'size': size,
                'query': {
                    'match': {
                        'topics': {
                            'query': topic,
                            'fuzziness': 'AUTO'
                        }
                    }
                }
            }

            response = client.search(
                index=self.opensearch_index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                result.pop('embedding', None)
                results.append(result)

            return results

        except Exception as e:
            print(f"Error performing topic search: {e}")
            return []

    def search_by_multiple_entities(
        self,
        entity_filters: List[Dict[str, str]],
        match_all: bool = False,
        size: int = 10
    ) -> List[Dict]:
        """
        Search for chunks containing multiple specific entities.

        Args:
            entity_filters: List of entity filters, each with 'type' and/or 'value'
            match_all: If True, requires all entities to be present. If False, matches any.
            size: Number of results to return

        Returns:
            List of matching documents with scores

        Example:
            # Find chunks mentioning both OpenAI and TensorFlow
            results = indexer.search_by_multiple_entities([
                {"type": "ORG", "value": "OpenAI"},
                {"type": "ORG", "value": "TensorFlow"}
            ], match_all=True)
        """
        try:
            client = self._get_opensearch_client()

            # Build nested queries for each entity filter
            entity_queries = []
            for entity_filter in entity_filters:
                must_conditions = []

                if 'type' in entity_filter:
                    must_conditions.append({
                        "match": {
                            "entities.type": entity_filter['type']
                        }
                    })

                if 'value' in entity_filter:
                    must_conditions.append({
                        "match": {
                            "entities.value": {
                                "query": entity_filter['value'],
                                "fuzziness": "AUTO"
                            }
                        }
                    })

                entity_queries.append({
                    'nested': {
                        'path': 'entities',
                        'query': {
                            'bool': {
                                'must': must_conditions
                            }
                        }
                    }
                })

            # Combine queries with must (AND) or should (OR)
            bool_clause = 'must' if match_all else 'should'

            search_body = {
                'size': size,
                'query': {
                    'bool': {
                        bool_clause: entity_queries
                    }
                }
            }

            response = client.search(
                index=self.opensearch_index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                result.pop('embedding', None)
                results.append(result)

            return results

        except Exception as e:
            print(f"Error performing multiple entity search: {e}")
            return []

    def search_combined(
        self,
        text_query: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_value: Optional[str] = None,
        topics: Optional[List[str]] = None,
        size: int = 10
    ) -> List[Dict]:
        """
        Combined search across text, entities, and topics.

        Args:
            text_query: Text to search in content
            entity_type: Entity type to filter by
            entity_value: Entity value to search for
            topics: List of topics to match
            size: Number of results to return

        Returns:
            List of matching documents with scores

        Example:
            # Find chunks about "neural networks" mentioning organizations
            results = indexer.search_combined(
                text_query="neural networks",
                entity_type="ORG",
                topics=["deep learning"]
            )
        """
        try:
            client = self._get_opensearch_client()

            must_conditions = []

            # Add text query
            if text_query:
                must_conditions.append({
                    'match': {
                        'content': {
                            'query': text_query,
                            'fuzziness': 'AUTO'
                        }
                    }
                })

            # Add entity filter
            if entity_type or entity_value:
                entity_must = []
                if entity_type:
                    entity_must.append({
                        "match": {
                            "entities.type": entity_type
                        }
                    })
                if entity_value:
                    entity_must.append({
                        "match": {
                            "entities.value": {
                                "query": entity_value,
                                "fuzziness": "AUTO"
                            }
                        }
                    })

                must_conditions.append({
                    'nested': {
                        'path': 'entities',
                        'query': {
                            'bool': {
                                'must': entity_must
                            }
                        }
                    }
                })

            # Add topic filters
            if topics:
                for topic in topics:
                    must_conditions.append({
                        'match': {
                            'topics': {
                                'query': topic,
                                'fuzziness': 'AUTO'
                            }
                        }
                    })

            search_body = {
                'size': size,
                'query': {
                    'bool': {
                        'must': must_conditions
                    }
                }
            }

            response = client.search(
                index=self.opensearch_index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                result.pop('embedding', None)
                results.append(result)

            return results

        except Exception as e:
            print(f"Error performing combined search: {e}")
            return []

    def health_check(self) -> Dict:
        """
        Check OpenSearch cluster health.

        Returns:
            Dictionary with cluster health information
        """
        try:
            client = self._get_opensearch_client()
            health = client.cluster.health()
            return health
        except Exception as e:
            print(f"Error checking cluster health: {e}")
            return {}

    def index_exists(self) -> bool:
        """
        Check if the configured index exists.

        Returns:
            bool: True if index exists, False otherwise
        """
        try:
            client = self._get_opensearch_client()
            return client.indices.exists(index=self.opensearch_index_name)
        except Exception as e:
            print(f"Error checking if index exists: {e}")
            return False
