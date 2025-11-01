# Semantic Chunker

A Python library for intelligent text chunking that preserves semantic boundaries while maintaining context through overlapping chunks. Supports both local files and AWS S3 objects.

## Features

- **Semantic Chunking**: Respects sentence and paragraph boundaries
- **Configurable Overlaps**: Maintains context between chunks with customizable overlap
- **Multiple File Types**: Supports `.txt` and `.ipynb` (Jupyter notebooks)
- **Local and S3 Support**: Read files from local filesystem or AWS S3
- **Flexible Chunking Parameters**: Configure chunk size, overlap, and minimum size
- **JSON Output**: Save chunks with metadata to JSON files

## Installation

```bash
# Basic installation
pip install -r requirements.txt

# Or install dependencies manually
pip install boto3>=1.28.0
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
│   ├── semantic_chunker.py    # Main chunking library
│   ├── main.py                # Example usage script
│   └── sample_text.txt        # Sample data
├── test/
│   └── test_chunker.py        # Test suite with S3 support
├── notebooks/
│   └── *.ipynb                # Jupyter notebooks
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Running Tests

### Run All Tests

```bash
python test/test_chunker.py
```

### Interactive Test Mode

```bash
python test/test_chunker.py interactive
```

This allows you to:
- Choose between local files or S3 URIs
- Configure chunk parameters interactively
- Test with custom files

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

```json
{
  "chunk_id": 0,
  "content": "The actual text content of this chunk...",
  "char_count": 847,
  "paragraph_start": 2
}
```

- **chunk_id**: Sequential chunk identifier
- **content**: The text content of the chunk
- **char_count**: Number of characters in the chunk
- **paragraph_start**: Index of the starting paragraph

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
- boto3 >= 1.28.0 (for S3 support)

## License

This project is part of a machine learning coursework repository.

## Contributing

Feel free to submit issues or pull requests for improvements.
