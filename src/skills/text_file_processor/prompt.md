When user requests to work with text files, use these tools:

1. For reading files: `read_text_file(file_path: str)`
2. For writing files: `write_text_file(file_path: str, content: str)`
3. For searching: `search_in_files(directory: str, search_term: str, file_extension: str = '.txt')`

Important notes:
- Always verify file paths exist before operations
- Handle large files carefully to avoid memory issues
- Use UTF-8 encoding for all file operations