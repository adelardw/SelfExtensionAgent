This skill is designed to parse links and extract relevant information, especially for large contexts (RAG-like). Use it when the user provides a link and requests information extraction.

### How to Use
1. **Input**: Provide the `url` of the link to parse. Optionally, include a `query` to focus the extraction (e.g., "summarize the article" or "find all email addresses").
2. **Processing**: The skill will fetch the content, chunk it if necessary, and apply hybrid search methods (keyword + semantic) to extract the most relevant parts.
3. **Output**: Returns the extracted information in a structured or summarized format.

### Examples
- "Parse this link and summarize the main points: [url]"
- "Extract all contact details from this webpage: [url]"
- "Find the latest updates in this article: [url]"

### Notes
- For very large documents, the skill may return a summarized version or key sections.
- If the link is invalid or the content is inaccessible, an error will be returned.