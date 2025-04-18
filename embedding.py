import lancedb
from pydantic import Field
from lancedb.pydantic import LanceModel, Vector
from typing import List, Optional
from lancedb.embeddings import get_registry 
import os
import numpy as np
import torch
from openai import OpenAI
from dotenv import load_dotenv
import time
from collections import OrderedDict

# Load environment variables
load_dotenv()

# Initialize OpenAI client for context generation
client = OpenAI()

# Create necessary directories with absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, "data/lancedb")
os.makedirs(db_path, exist_ok=True)
# LanceDB will handle persistence

# Connect to LanceDB using absolute path
print(f"Connecting to LanceDB at: {db_path}")
db = lancedb.connect(db_path)

# We use a hybrid approach: OpenAI for context generation and Qwen local model for embeddings
# openai_func is only used for context generation, not for embeddings
openai_func = get_registry().get("openai").create(name="text-embedding-3-large")

# Default Qwen model for embeddings
DEFAULT_QWEN_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"


def create_correct_schema(vector_dimension):
    """Create a PyArrow schema with the correct nullability settings for LanceDB"""
    import pyarrow as pa
    
    # Create a schema where fields are nullable to match our data
    metadata_schema = pa.struct([
        pa.field('context', pa.string(), nullable=True),
        pa.field('filename', pa.string(), nullable=True),
        pa.field('page_numbers', pa.list_(pa.int64()), nullable=True),
        pa.field('title', pa.string(), nullable=True)
        ])
    
    return pa.schema([
        pa.field('text', pa.string(), nullable=True),
        pa.field('vector', pa.list_(pa.float64(), vector_dimension), nullable=False),
        pa.field('metadata', metadata_schema, nullable=True),
    ])

################################################################################
# Context Generation Using OpenAI
################################################################################

def build_contextual_prompt(whole_document: str, chunk: str) -> str:
    """Builds a prompt for OpenAI to generate context for a chunk as bullet points, focusing on how the chunk is situated in the rest of the document."""
    prompt = f"""
    <document>
    {whole_document}
    </document>

    Below is a chunk from the document.

    Write 3-4 bullet points (not a paragraph, and do not repeat the chunk text, do not use the word chunk at all in your response) that describe how this chunk is situated within the rest of the document. Focus on:
    - What the chunk is about in relation to the whole document
    - What came before and after (if relevant)
    - How it connects to the document's main ideas or sections
    - Any transitions, summaries, or context that helps place it
    
    Use concise, technical language with minimal words per bullet; focus on clarity, structure, and relevance to the document’s flow without repeating content. Avoid full sentences unless necessary for context.
    
    Example output from a chunk:
    - Explains standard encoder-decoder structure in sequence models
    - Introduces how the Transformer fits this framework
    - Prepares reader for detailed explanation of Transformer components
    
    Chunk:
    """
    prompt += f"""
    {chunk}
    """
    prompt += """
    Bullet points:
-"""
    return prompt.strip()


def generate_chunk_context_openai(
    chunk: str,
    whole_document: str,
    document_path: str,
    openai_chat_model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_retries: int = 3
) -> Optional[str]:
    """Generate a context for a chunk using OpenAI without caching."""
    # Get the chunk text
    chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
    chunk_preview = chunk_text[:50] + '...' if len(chunk_text) > 50 else chunk_text
    print(f"  - Generating context for chunk: '{chunk_preview}'")
    
    prompt = build_contextual_prompt(whole_document, chunk_text)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=openai_chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=128
            )
            context = response.choices[0].message.content.strip()
            return context
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to generate context after {max_retries} attempts: {str(e)}")
                return None
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff

def build_contextual_chunk(chunk: str, context_str: str) -> str:
    """Prepends the LLM-generated context to the chunk text for embedding."""
    return f"{context_str}\n{chunk}".strip()

################################################################################
# Qwen Embedding Functions
################################################################################

def load_qwen_embedding_model(model_name_or_path: str, device: str = "cpu"):
    """Load the Qwen embedding model and tokenizer."""
    try:
        from transformers import AutoTokenizer, AutoModel
        print(f"Loading Qwen model: {model_name_or_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
        
        # Move model to the appropriate device
        model.to(device)
        
        # Handle multi-GPU setup if available and not on CPU
        if device != "cpu" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)
            
        model.eval()  # Set to evaluation mode
        print("Qwen model loaded successfully.")
        return tokenizer, model
        
    except Exception as e:
        print(f"Error loading Qwen model: {str(e)}")
        raise
def compute_embeddings_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute embeddings for a batch of texts using parallel processing if available.
    """

    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"\nProcessing {len(texts)} texts in {total_batches} batches of size {batch_size}")
    
    all_embeddings = []
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch = texts[start_idx:end_idx]
        
        print(f"\n  - Batch {batch_idx + 1}/{total_batches}:")
        print(f"    Processing items {start_idx + 1}-{end_idx} of {len(texts)}")
        
        print(f"    Tokenizing texts... ({start_idx + 1}-{end_idx}/{len(texts)} chunks)")
        # Process each chunk individually to provide detailed progress
        for i, text in enumerate(batch):
            print(f"      Processing chunk {start_idx + i + 1}/{len(texts)}", flush=True)
            
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"    Computing embeddings...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get embeddings from the last hidden state using mean pooling
        print(f"    Extracting embeddings from model output...")
        # Use mean pooling over sequence length (dim=1)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            # Mask out padding tokens before taking mean
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            masked = last_hidden * mask
            summed = torch.sum(masked, dim=1)
            counts = torch.clamp(torch.sum(attention_mask, dim=1).unsqueeze(-1), min=1e-9)
            batch_embeddings = (summed / counts).cpu().numpy()
        else:
            # Fallback to simple mean if no attention mask
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        all_embeddings.append(batch_embeddings)
        print(f"    Batch complete. Shape: {batch_embeddings.shape}")
        
        # Print some stats about the embeddings
        print(f"    Embedding stats - Mean: {batch_embeddings.mean():.3f}, Std: {batch_embeddings.std():.3f}")
    
    print("\nConcatenating all batches...")
    final_embeddings = np.vstack(all_embeddings)
    print(f"Final shape: {final_embeddings.shape}")
    
    return final_embeddings

def embed_chunks_with_qwen(chunks: List[str], tokenizer, model, batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """
    Embeds a list of chunks using parallel processing and efficient batching.
    Returns an array of shape (num_chunks, embedding_dim).
    """
    print("\n" + "="*80)
    print("Starting Qwen Embedding Generation")
    print("="*80)
    print(f"Total chunks to process: {len(chunks)}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Average chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} characters")
    
    # Process embeddings in parallel batches
    embeddings = compute_embeddings_batch(
        chunks,
        tokenizer,
        model,
        device=device,
        batch_size=batch_size
    )
    
    print("\n" + "="*80)
    print(f"Embedding Generation Complete")
    print("="*80)
    print(f"Output shape: {embeddings.shape}")
    print(f"Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    return embeddings

################################################################################
# LanceDB Models
################################################################################

class ChunkMetadata(LanceModel):
    # Give filename a default so it’s not "required" in Arrow
    filename: str = ""
    # Use default_factory for lists so the schema is strictly non-null
    page_numbers: List[int] = Field(default_factory=list)
    title: str = ""
    context: str = ""


# We'll determine the vector dimension dynamically when we initialize the table
# This way it will work with any model and dimensions

class Chunks(LanceModel):
    text: str  # This will be our source field
    vector: Vector(1)  # Placeholder, we'll use the actual dimension when creating the table
    metadata: ChunkMetadata

def embed_document(chunks, existing_table=None, document_path=None):
    """Embeds chunks into LanceDB using the contextual embedding approach.
    
    Args:
        chunks: List of document chunks from docling
        existing_table: Existing LanceDB table (optional)
        document_path: Path to the original document 
    
    Returns:
        LanceDB table with embedded chunks
    """
    if not document_path:
        raise ValueError("document_path is required for contextual embeddings")
    
    # Use the existing table if provided
    if existing_table is not None:
        print("Using existing table for embedding")
        table = existing_table
    else:
        # Create a sample embedding to determine dimensions
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model = load_qwen_embedding_model(DEFAULT_QWEN_MODEL, device=device)
        sample_text = "Sample text for dimension detection"
        sample_emb = embed_chunks_with_qwen([sample_text], tokenizer, model, device=device)[0]
        vector_dimension = len(sample_emb)
        print(f"Detected Qwen embedding dimension: {vector_dimension}")
        
        # Create a custom schema with the right nullability constraints
        table_schema = create_correct_schema(vector_dimension)

        # Check if table exists first
        try:
            table = db.open_table("docling")
            print(f"Opened existing table with {table.count_rows()} rows")
        except Exception:
            # Create a new table if it doesn't exist
            table = db.create_table("docling", schema=table_schema, mode="create")
            print(f"Created new table with vector dimension: {vector_dimension}")
            
    # Load the model if not already loaded
    if 'tokenizer' not in locals() or 'model' not in locals():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model = load_qwen_embedding_model(DEFAULT_QWEN_MODEL, device=device)
            
    return embed_document_contextual(chunks, table, document_path, tokenizer, model)

def embed_document_contextual(chunks, table, document_path, tokenizer, model):
    """Embeds chunks with contextual retrieval approach.
    
    Uses OpenAI to generate context for each chunk and Qwen for embeddings.
    
    Args:
        chunks: List of document chunks
        table: LanceDB table
        document_path: Path to the original document
        vector_dimension: The dimension of the vectors to generate (determined dynamically)
    
    Returns:
        LanceDB table with contextually embedded chunks
    """
    # Get all the text to reconstruct the whole document
    whole_document = "\n\n".join([chunk.text for chunk in chunks])
    
    # Initialize arrays for embeddings and processed chunks
    contexts = []
    all_contextualized_chunks = []
    processed_chunks = []
        
    # Process each chunk to generate contextual embeddings
    total_chunks = len(chunks)
    print(f"Starting to process {total_chunks} chunks for document: {os.path.basename(document_path)}")
    for idx, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {idx}/{total_chunks} ({idx/total_chunks*100:.1f}%)", flush=True)
        # Generate context for this chunk
        context = generate_chunk_context_openai(
            chunk=chunk,
            whole_document=whole_document,
            document_path=document_path,
            temperature=0.7
        )
        print(f"  - Context generation {'completed' if context else 'failed'} for chunk {idx}")
        contexts.append(context)
        contextualized_chunk = build_contextual_chunk(chunk.text, context)
        all_contextualized_chunks.append(contextualized_chunk)
        
        # Create metadata object properly
        metadata = ChunkMetadata(
            filename=(chunk.meta.origin.filename or "") if chunk.meta and chunk.meta.origin else "",
            page_numbers=[
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in (chunk.meta.doc_items or [])
                        for prov in (item.prov or [])
                        if prov.page_no is not None
                    )
                )
            ] if chunk.meta and chunk.meta.doc_items else [],
            title=(chunk.meta.headings[0] if chunk.meta.headings else "") if chunk.meta and chunk.meta.headings else "",
            context=context or "",  # Fallback to empty string if context is None
        )
        
        # Create metadata dict from Pydantic model
        metadata_dict = OrderedDict({
            "context": metadata.context or "",
            "filename": metadata.filename or "",
            "page_numbers": metadata.page_numbers or [],
            "title": metadata.title or "",
        })
        processed_chunks.append({
            "text": chunk.text or "",
            "metadata": metadata_dict,
        })
    
    # Compute embeddings in one batch for all contextualized chunks
    print(f"Computing embeddings for {len(all_contextualized_chunks)} contextualized chunks...")
    all_embeddings = embed_chunks_with_qwen(
        all_contextualized_chunks, tokenizer, model, batch_size=32
    )
    print(f"✅ Completed embedding generation with {len(all_embeddings)} vectors created")
    
    # Add embeddings to the processed chunks
    print(f"Adding embeddings to {len(processed_chunks)} processed chunks...")
    for i, processed_chunk in enumerate(processed_chunks):
        processed_chunk["vector"] = all_embeddings[i].tolist()
    print(f"✅ Completed adding embeddings to {len(processed_chunks)} chunks")
    
    # Add to LanceDB table with schema verification
    if processed_chunks:
        try:
            # Add to table
            print(f"Adding {len(processed_chunks)} chunks to LanceDB table...")
            table.add(processed_chunks)
            print(f"✅ Successfully added {len(processed_chunks)} chunks to LanceDB")
            
            row_count = table.count_rows()
            print(f"🔎 Table now contains {row_count} rows in total")
            
            print("✅ Using pure vector search with cosine similarity")
            
            # Verify the table is working by getting basic info
            try:
                row_count = table.count_rows()
                print(f"✅ Verified table functionality - table contains {row_count} rows")
            except Exception as e:
                print(f"Warning: Table verification failed: {e}, but we'll continue anyway")

        except Exception as e:
            print(f"Error adding to table: {e}")
            # Return the existing table so the application can continue
            return existing_table

    # Always return a valid table reference
    return table

def embed_document_simple(chunks, existing_table=None, document_path=None):
    """Embeds chunks using OpenAI sentence-level embeddings into a dedicated LanceDB table."""
    if not document_path:
        raise ValueError("document_path is required for embeddings")
    # Prepare data for embedding
    texts = [chunk.text or "" for chunk in chunks]
    print(f"⏳ Generating OpenAI embeddings for {len(texts)} chunks...")
    response = client.embeddings.create(model="text-embedding-3-large", input=texts)
    embeddings = [item.embedding for item in response.data]
    # Build processed records
    processed = []
    for chunk, vector in zip(chunks, embeddings):
        page_numbers = sorted(
            prov.page_no
            for item in (chunk.meta.doc_items or [])
            for prov in (item.prov or []) if prov.page_no is not None
        )
        metadata = {
            "context": "",
            "filename": (chunk.meta.origin.filename or "") if chunk.meta and chunk.meta.origin else "",
            "page_numbers": page_numbers,
            "title": (chunk.meta.headings[0] if chunk.meta.headings else "") if chunk.meta and chunk.meta.headings else "",
        }
        processed.append({"text": chunk.text or "", "vector": vector, "metadata": metadata})
    # Upsert into a simple embeddings table
    table_name = "docling_simple"
    try:
        table = db.open_table(table_name)
        print(f"🔄 Appending {len(processed)} embeddings to {table_name}")
        table.add(processed)
    except Exception:
        print(f"🆕 Creating table {table_name} with {len(processed)} entries")
        table = db.create_table(table_name, processed, mode="create")
    return table
