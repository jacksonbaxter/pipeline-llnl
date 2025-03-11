from docling.chunking import HybridChunker
from utils.tokenizer import OpenAITokenizerWrapper

# Ensure that the tokenizer is initialized
def get_tokenizer():
    tokenizer = tokenizer = OpenAITokenizerWrapper()
    if tokenizer is None:
        raise ValueError("Tokenizer failed to initialize. Check OpenAITokenizerWrapper.")
    return tokenizer

tokenizer = get_tokenizer()
MAX_TOKENS = 8191

def chunk_document(document):
    """Chunks a document using HybridChunker and returns a list of chunks."""
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
    return list(chunker.chunk(dl_doc=document))  # Return chunked text
