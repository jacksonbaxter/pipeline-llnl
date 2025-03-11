from docling.document_converter import DocumentConverter

def extract_document(file_path):
    """Extracts text from the document and returns a Docling document object."""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document  # Return the extracted document
