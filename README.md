# Semantic Chunker

A Python library for intelligent text chunking that preserves semantic boundaries while maintaining context through overlapping chunks. Supports both local files and AWS S3 objects.

## Features

- **Semantic Chunking**: Respects sentence and paragraph boundaries
- **Configurable Overlaps**: Maintains context between chunks with customizable overlap
- **Multiple File Types**: Supports `.txt` and `.ipynb` (Jupyter notebooks)
- **Local and S3 Support**: Read files from local filesystem or AWS S3
- **Entity & Topic Extraction**: Extract entities (people, organizations, locations, dates) and topics using Claude Sonnet AI
- **Vector Embeddings**: Generate embeddings using AWS Bedrock Titan text model
- **OpenSearch Integration**: Index chunks with text, vector, and hybrid search capabilities
- **RAG (Retrieval-Augmented Generation)**: Ask questions and get AI-powered answers with source citations
- **Flexible Chunking Parameters**: Configure chunk size, overlap, and minimum size
- **JSON Output**: Save chunks with metadata to JSON files

## Installation

```bash
# Basic installation
pip install -r requirements.txt

# Or install dependencies manually
pip install boto3>=1.28.0 anthropic>=0.39.0 python-dotenv>=1.0.0 opensearch-py>=2.3.0
```

### API Key Setup (for Entity/Topic Extraction)

To use entity and topic extraction features, you need an Anthropic API key:

1. Get your API key from: https://console.anthropic.com/settings/keys
2. Copy `.env.example` to `.env`
3. Add your API key to `.env`:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Quick Start

### Local Files

```python
from src.semantic_chunker import SemanticChunker

# Initialize chunker
chunker = SemanticChunker(
    chunk_size=1000,      # Target chunk size in characters
    chunk_overlap=200,    # Overlap between chunks
    min_chunk_size=100    # Minimum chunk size
)

# Chunk a local file
chunks = chunker.chunk_file("path/to/document.txt")

# Print summary
chunker.print_chunk_summary(chunks)

# Save to JSON
chunker.save_chunks(chunks, "output.json")
```

### AWS S3 Files

```python
from src.semantic_chunker import SemanticChunker

# Option 1: Use default AWS credentials
# (from ~/.aws/credentials, environment variables, or IAM role)
chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_file("s3://my-bucket/path/to/document.txt")

# Option 2: Provide credentials explicitly
chunker = SemanticChunker(
    chunk_size=1000,
    chunk_overlap=200,
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    aws_region="us-east-1"
)
chunks = chunker.chunk_file("s3://my-bucket/data/notebook.ipynb")
```

### Entity and Topic Extraction

Extract entities (people, organizations, locations, dates) and topics from chunks using Claude Sonnet:

```python
from src.semantic_chunker import SemanticChunker

# Enable entity/topic extraction
chunker = SemanticChunker(
    chunk_size=1000,
    chunk_overlap=200,
    extract_entities_topics=True  # Enable AI-powered extraction
)

# Chunk a file and extract entities/topics
chunks = chunker.chunk_file("document.txt")

# Access extracted data
for chunk in chunks:
    print(f"Chunk {chunk['chunk_id']}")
    print(f"Entities: {chunk['entities']}")
    # Example: [{"type": "PERSON", "value": "John Smith"}, {"type": "ORG", "value": "OpenAI"}]

    print(f"Topics: {chunk['topics']}")
    # Example: ["machine learning", "artificial intelligence"]
    print()

# Save chunks with entities/topics
chunker.save_chunks(chunks, "chunks_with_entities.json")
```

**Note**: Entity/topic extraction requires an Anthropic API key and will make API calls for each chunk, which may incur costs.

### OpenSearch Integration with Vector Embeddings

Index chunks to AWS OpenSearch for text, vector, and hybrid search capabilities:

```python
from src.semantic_chunker import SemanticChunker
from src.opensearch_indexer import OpenSearchIndexer

# Step 1: Create chunks with entities/topics
chunker = SemanticChunker(
    chunk_size=1000,
    chunk_overlap=200,
    extract_entities_topics=True
)
chunks = chunker.chunk_file("document.txt")

# Step 2: Initialize OpenSearch indexer
indexer = OpenSearchIndexer(
    opensearch_host="your-domain.region.es.amazonaws.com",
    opensearch_index_name="semantic_chunks",
    bedrock_region="us-east-1"
)

# Step 3: Index chunks (generates embeddings automatically)
result = indexer.index_chunks(chunks)
print(f"Indexed {result['success']} chunks")

# Step 4: Perform searches
# Text search (BM25)
text_results = indexer.search_text("machine learning", size=5)

# Vector search (kNN similarity)
vector_results = indexer.search_vector("deep learning frameworks", size=5)

# Hybrid search (combined text + vector)
hybrid_results = indexer.search_hybrid(
    "Python data science",
    size=5,
    text_weight=0.5,
    vector_weight=0.5
)

# Access results
for result in hybrid_results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
    print(f"Entities: {result.get('entities', [])}")
    print(f"Topics: {result.get('topics', [])}")
```

**Entity and Topic Based Search:**

```python
from src.opensearch_indexer import OpenSearchIndexer

indexer = OpenSearchIndexer()

# Search by entity type (find all chunks mentioning organizations)
org_results = indexer.search_by_entity(entity_type="ORG", size=10)

# Search by entity value (find chunks mentioning "Python")
python_results = indexer.search_by_entity(entity_value="Python", size=10)

# Search by both type and value (find organizations named "OpenAI")
specific_results = indexer.search_by_entity(
    entity_type="ORG",
    entity_value="OpenAI",
    size=10
)

# Search by topic
ml_results = indexer.search_by_topic("machine learning", size=10)

# Search for multiple entities (chunks mentioning both OpenAI AND TensorFlow)
multi_results = indexer.search_by_multiple_entities([
    {"type": "ORG", "value": "OpenAI"},
    {"type": "ORG", "value": "TensorFlow"}
], match_all=True, size=10)

# Combined search (text + entities + topics)
combined_results = indexer.search_combined(
    text_query="neural networks",
    entity_type="ORG",
    topics=["deep learning", "AI"],
    size=10
)

# Access entity information
for result in combined_results:
    print(f"Content: {result['content'][:100]}...")
    print(f"Organizations: {[e['value'] for e in result['entities'] if e['type'] == 'ORG']}")
    print(f"Topics: {result['topics']}")
```

**Setup Requirements:**
1. **Create OpenSearch index** (one-time setup):
   ```bash
   python scripts/create_opensearch_index.py
   ```

2. **Configure environment variables** in `.env`:
   ```bash
   OPENSEARCH_HOST=your-domain.region.es.amazonaws.com
   OPENSEARCH_PORT=443
   OPENSEARCH_INDEX_NAME=semantic_chunks
   BEDROCK_REGION=us-east-1
   TITAN_MODEL_ID=amazon.titan-embed-text-v1
   ```

3. **AWS Permissions Required:**
   - OpenSearch: Domain access for read/write
   - Bedrock: `bedrock:InvokeModel` permission for Titan embeddings

**Search Capabilities:**
- **Text Search**: Traditional keyword search using BM25 algorithm
- **Vector Search**: Semantic similarity search using embeddings
- **Hybrid Search**: Combines text and vector search with configurable weights
- **Entity Search**: Filter by entity type (PERSON, ORG, LOC, DATE) or value
- **Topic Search**: Find chunks by extracted topics
- **Multi-Entity Search**: Find chunks mentioning multiple specific entities
- **Combined Search**: Mix text queries with entity/topic filters

### RAG (Retrieval-Augmented Generation)

Ask questions and get AI-powered answers based on your indexed documents with source citations:

```python
from src.rag_generator import RAGGenerator

# Initialize RAG system
rag = RAGGenerator()

# Ask a single question
result = rag.ask("What is machine learning?", num_chunks=5)

print("Answer:", result['answer'])
print("\nSources:", result['sources'])
print(f"\nRetrieved {len(result['chunks'])} relevant chunks")

# Multi-turn conversation with history
result1 = rag.chat("What is deep learning?")
result2 = rag.chat("How does it differ from traditional ML?")
result3 = rag.chat("Give me some examples")

# Each response builds on previous context
print(result3['answer'])

# View conversation history
history = rag.get_history()
print(f"Conversation has {len(history)} turns")

# Clear history when starting new topic
rag.clear_history()
```

**RAG Features:**

- **Automatic Search Strategy**: Intelligently selects best search method (text/vector/hybrid/entity) based on query
- **Query Enhancement**: Uses LLM to improve queries before retrieval for better results
- **Conversation History**: Maintains context across multiple questions
- **Source Citations**: Includes chunk IDs, scores, entities, and topics for transparency
- **Configurable Retrieval**: Adjust number of chunks, temperature, and other parameters

**Configuration** (in `.env`):

```bash
# Claude model for generation
BEDROCK_CLAUDE_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Number of chunks to retrieve (more = more context, slower)
RAG_MAX_CHUNKS=5

# Temperature (0.0 = deterministic, 1.0 = creative)
RAG_TEMPERATURE=0.7

# Max conversation turns to remember
RAG_MAX_HISTORY=10
```

**Interactive Mode:**

```bash
# Launch interactive RAG CLI
python scripts/interactive_rag.py
```

Available commands in interactive mode:
- `/help` - Show help
- `/clear` - Clear conversation history
- `/history` - View conversation
- `/stats` - Show system stats
- `/chunks N` - Set number of chunks
- `/temp N` - Set temperature
- `/quit` - Exit

**Example Workflow:**

```bash
# 1. Index your documents
python src/example_opensearch.py

# 2. Ask questions
python src/example_rag.py

# 3. Or use interactive mode
python scripts/interactive_rag.py
```

## AWS Credentials Setup

The chunker supports multiple methods for AWS authentication, tried in this order:

### 1. Explicit Credentials (Constructor)

```python
chunker = SemanticChunker(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    aws_session_token="YOUR_SESSION_TOKEN",  # Optional
    aws_region="us-east-1"  # Optional, defaults to us-east-1
)
```

### 2. Environment Variables

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 3. AWS Credentials File

Create or edit `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

Create or edit `~/.aws/config`:

```ini
[default]
region = us-east-1
```

### 4. IAM Role

If running on AWS infrastructure (EC2, Lambda, ECS, etc.), the chunker will automatically use the IAM role attached to the service.

## Supported File Formats

### Text Files (`.txt`)
Plain text files are read directly and chunked semantically.

### Jupyter Notebooks (`.ipynb`)
Notebook cells are extracted with tags:
- `[MARKDOWN]` - Markdown cells
- `[CODE]` - Code cells
- `[OUTPUT]` - Cell outputs

This preserves the structure while allowing semantic chunking across all content.

## Usage Examples

### Basic Text Chunking

```python
from src.semantic_chunker import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk_file("document.txt")

# Access chunk data
for chunk in chunks:
    print(f"Chunk {chunk['chunk_id']}")
    print(f"Size: {chunk['char_count']} characters")
    print(f"Content: {chunk['content'][:100]}...")
    print()
```

### Custom Chunk Parameters

```python
# Smaller chunks with more overlap
chunker = SemanticChunker(
    chunk_size=500,       # Smaller chunks
    chunk_overlap=150,    # More overlap for context
    min_chunk_size=50     # Allow smaller chunks
)

# Larger chunks with less overlap
chunker = SemanticChunker(
    chunk_size=2000,      # Larger chunks
    chunk_overlap=100,    # Less overlap
    min_chunk_size=200    # Enforce minimum size
)
```

### S3 with Different Regions

```python
# US West region
chunker = SemanticChunker(aws_region="us-west-2")
chunks = chunker.chunk_file("s3://my-west-bucket/file.txt")

# EU region
chunker = SemanticChunker(aws_region="eu-west-1")
chunks = chunker.chunk_file("s3://my-eu-bucket/data.ipynb")
```

## Project Structure

```
Machine_Learning/
├── src/
│   ├── semantic_chunker.py        # Main chunking library
│   ├── entity_topic_extractor.py  # Entity/topic extraction using Claude Sonnet
│   ├── opensearch_indexer.py      # OpenSearch integration with Bedrock embeddings
│   ├── rag_generator.py           # RAG system with Claude for Q&A
│   ├── example_opensearch.py      # OpenSearch workflow demo
│   ├── example_rag.py             # RAG system demo
│   ├── main.py                    # Basic usage example
│   └── sample_text.txt            # Sample data
├── scripts/
│   ├── create_opensearch_index.py # Create OpenSearch index with proper mappings
│   └── interactive_rag.py         # Interactive RAG CLI
├── test/
│   ├── test_chunker.py            # Chunking tests
│   ├── test_opensearch.py         # OpenSearch integration tests
│   └── test_rag.py                # RAG system tests
├── notebooks/
│   └── *.ipynb                    # Jupyter notebooks
├── requirements.txt               # Python dependencies
├── .env.example                   # Example environment configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Running Tests

### Chunking Tests

```bash
# Run all chunking tests
python test/test_chunker.py

# Interactive test mode
python test/test_chunker.py interactive
```

### OpenSearch Tests

```bash
# Run OpenSearch integration tests
python test/test_opensearch.py
```

**Note**: OpenSearch tests require:
- OpenSearch cluster configured and accessible
- AWS credentials set up
- Index created (run `python scripts/create_opensearch_index.py` first)

### RAG Tests

```bash
# Run RAG system tests
python test/test_rag.py
```

**Note**: RAG tests require:
- OpenSearch configured with indexed data
- AWS Bedrock Claude access enabled
- All previous requirements met

### Full Workflow Demos

```bash
# OpenSearch indexing workflow
python src/example_opensearch.py

# RAG Q&A demonstration
python src/example_rag.py

# Interactive RAG CLI
python scripts/interactive_rag.py
```

**RAG Example Features:**
- Single question answering
- Multi-turn conversations
- Query enhancement demos
- Automatic search strategy selection
- Interactive CLI with commands

### Example Usage Script

```bash
python src/main.py
```

## Error Handling

The chunker provides clear error messages for common issues:

### S3 Errors

```python
try:
    chunks = chunker.chunk_file("s3://bucket/key")
except FileNotFoundError:
    print("S3 bucket or object not found")
except PermissionError:
    print("Access denied - check your AWS permissions")
except RuntimeError as e:
    print(f"S3 error: {e}")
```

### Local File Errors

```python
try:
    chunks = chunker.chunk_file("nonexistent.txt")
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Unsupported file type: {e}")
```

## Chunk Output Format

Each chunk is a dictionary with the following structure:

### Basic Format (without entity/topic extraction)

```json
{
  "chunk_id": 0,
  "content": "The actual text content of this chunk...",
  "char_count": 847,
  "paragraph_start": 2
}
```

### With Entity/Topic Extraction

```json
{
  "chunk_id": 0,
  "content": "The actual text content of this chunk...",
  "char_count": 847,
  "paragraph_start": 2,
  "entities": [
    {"type": "PERSON", "value": "John Smith"},
    {"type": "ORG", "value": "OpenAI"},
    {"type": "LOC", "value": "San Francisco"},
    {"type": "DATE", "value": "2024"}
  ],
  "topics": [
    "machine learning",
    "artificial intelligence",
    "neural networks"
  ]
}
```

**Fields:**
- **chunk_id**: Sequential chunk identifier
- **content**: The text content of the chunk
- **char_count**: Number of characters in the chunk
- **paragraph_start**: Index of the starting paragraph
- **entities** (optional): Extracted named entities with type and value
- **topics** (optional): Main topics/themes identified in the chunk

## Development

### Debug Mode

The main script includes debug output:

```python
python src/main.py
```

Output includes:
- Chunker configuration
- Number of chunks generated
- Chunk size statistics
- Preview of first chunks

### VSCode Debugging

The project includes VSCode debug configurations:
- Press `F5` to debug `main.py`
- Or select "Python: Debug test_chunker.py"

## Requirements

- Python 3.7+
- boto3 >= 1.28.0 (for S3 support and Bedrock embeddings)
- anthropic >= 0.39.0 (for entity/topic extraction)
- python-dotenv >= 1.0.0 (for environment variable management)
- opensearch-py >= 2.3.0 (for OpenSearch integration)

## AWS Services Used

- **S3**: File storage and retrieval
- **Bedrock**: Titan text embeddings for vector search
- **OpenSearch**: Text, vector, and hybrid search with kNN support

## License

This project is part of a machine learning coursework repository.

## Contributing

Feel free to submit issues or pull requests for improvements.
