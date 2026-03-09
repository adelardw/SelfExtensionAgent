This skill parses links provided by the user to extract relevant information, handling large contexts effectively (RAG-like approach). It supports hybrid search methods to ensure accurate and up-to-date information retrieval.

### When to Use
- When the user provides a link and requests information extraction.
- When the context is too large for direct processing, requiring chunking or summarization.
- When hybrid search methods (e.g., combining keyword and semantic search) are needed for better results.

### Inputs
- `url`: The link to parse and extract information from.
- `query` (optional): A specific query to focus the extraction (e.g., "extract the main points" or "find contact details").

### Outputs
- Extracted and summarized information from the link.
- Structured data (if applicable, e.g., tables, lists).
- Error messages if parsing fails.