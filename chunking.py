from docling.chunking import HybridChunker
from utils.tokenizer import OpenAITokenizerWrapper

# Ensure that the tokenizer is initialized
def get_tokenizer():
    tokenizer = tokenizer = OpenAITokenizerWrapper()
    if tokenizer is None:
        raise ValueError("Tokenizer failed to initialize. Check OpenAITokenizerWrapper.")
    return tokenizer

tokenizer = get_tokenizer()
# Reduced chunk size for more precise splits
MAX_TOKENS = 512
# Overlap tokens between chunks to cover boundary context
OVERLAP_TOKENS = 50

def chunk_document(document):
    """Chunks a document using HybridChunker and returns a list of chunks."""
    # Use overlap_tokens to include context across chunk boundaries
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
        merge_peers=True,
    )
    return list(chunker.chunk(dl_doc=document))  # Return chunked text