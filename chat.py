import streamlit as st
import torch
import lancedb
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import fitz  
import os
import shutil
import json
import datetime
from streamlit_pdf_viewer import pdf_viewer
from extraction import extract_document
from chunking import chunk_document
from embedding import embed_document, Chunks, embed_chunks_with_qwen, load_qwen_embedding_model, DEFAULT_QWEN_MODEL

def check_api_key():
    """Check if OpenAI API key exists in environment variables."""
    return os.getenv("OPENAI_API_KEY") is not None

def save_api_key(api_key):
    """Save OpenAI API key to .env file."""
    env_path = ".env"
    
    # Read existing .env content
    env_content = ""
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_content = f.read()
    
    # Check if OPENAI_API_KEY already exists in the file
    if "OPENAI_API_KEY=" in env_content:
        # Replace existing key
        lines = env_content.splitlines()
        updated_lines = []
        for line in lines:
            if line.startswith("OPENAI_API_KEY="):
                updated_lines.append(f"OPENAI_API_KEY={api_key}")
            else:
                updated_lines.append(line)
        env_content = "\n".join(updated_lines)
    else:
        # Append new key
        if env_content and not env_content.endswith("\n"):
            env_content += "\n"
        env_content += f"OPENAI_API_KEY={api_key}"
    
    # Write back to .env file
    with open(env_path, "w") as f:
        f.write(env_content)
    
    # Update environment variable in current session
    os.environ["OPENAI_API_KEY"] = api_key
    return True

# Load environment variables
load_dotenv()

# Initialize OpenAI client (only if API key exists)
client = None
if check_api_key():
    client = OpenAI()

# Define PDF directory - update this to where your PDFs are stored
PDF_DIR = "data/pdfs"

table = None

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(PDF_DIR, exist_ok=True)

# Make sure directories exist at app startup
ensure_directories()


# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    # Use absolute path for LanceDB to avoid path resolution issues
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "data/lancedb")
    print(f"Connecting to LanceDB at: {db_path}")
    db = lancedb.connect(db_path)
    
    # Normal path for first initialization
    try:
        # Try to open the existing table
        table = db.open_table("docling")
        # Verify that the table exists and is working properly
        try:
            # Just count the rows to verify the table is working
            row_count = table.count_rows()
            print(f"✅ Table opened successfully with {row_count} rows")
        except Exception as e:
            # Log the error but continue
            print(f"Info: Table access test: {e}")
        return table
    except Exception:
        # Create a new table if it doesn't exist
        print("Table 'docling' doesn't exist, creating a new one")
        db.create_table("docling", schema=Chunks, mode="create")
        table = db.open_table("docling")
        return table
def get_table():
    global table
    if table is None:
        table = init_db()
    return table

    
table = get_table()

def process_document(file_path, file_name):
    """Run extraction, chunking, and embedding using existing modules.
    For PDFs, also save a copy in the PDF_DIR if needed.
    """
    global table
    
    # Only copy the file if it's not already in the PDF_DIR
    pdf_dest_path = os.path.join(PDF_DIR, file_name)
    if file_path != pdf_dest_path and file_name.lower().endswith('.pdf'):
        shutil.copyfile(file_path, pdf_dest_path)
        print(f"✅ {file_name} saved to {PDF_DIR}")
    
    document = extract_document(file_path)
    print(f"✅ {file_name} extracted!")
    chunks = chunk_document(document)
    print(f"✅ {file_name} chunked!")
    
    # Use the table from init_db and update our table variable with the returned one
    # Pass document path for contextual embeddings
    new_table = embed_document(chunks, existing_table=table, document_path=file_path)
    
    # Store the table in the global variable
    if new_table is not None:
        table = new_table
        
    print(f"✅ {file_name} embedded!")
    return f"✅ {file_name} processed and stored successfully."

def cosine_similarity_search(query: str, tokenizer, model, chunk_embs: np.ndarray, chunk_texts: list, top_k: int = 5, device: str = "cpu"):
    """
    Search local chunk embeddings for the chunks most similar to the query using Qwen embeddings.
    Returns a list of (chunk_text, similarity_score).
    """
    # 1. Encode the query using Qwen
    query_emb = embed_chunks_with_qwen([query], tokenizer, model, device=device)[0]
    
    # 2. Compute dot product
    dot_products = chunk_embs @ query_emb
    
    # 3. Normalize for cosine similarity
    chunk_norms = np.linalg.norm(chunk_embs, axis=1)
    query_norm = np.linalg.norm(query_emb)
    similarities = dot_products / (chunk_norms * query_norm + 1e-10)
    
    # 4. Sort and pick top_k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(chunk_texts[i], float(similarities[i])) for i in top_indices]
    return results

def get_context(query: str, table, num_results: int = 3) -> tuple:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    # Use the provided table
    search_table = table
    print("Using table for search")
        
    # Initialize contexts to handle the case where results is None
    contexts = []
    results = None
    
    try:
        # Use pure vector search with cosine similarity
        print("Performing vector search with cosine similarity...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model = load_qwen_embedding_model(DEFAULT_QWEN_MODEL, device=device)
        # query_vector = embed_chunks_with_qwen([query], tokenizer, model, device=device)[0]
        
        # # Search and get results with similarity scores
        # results = search_table.search(query_vector).limit(num_results).to_pandas()
        
        # Get all data from table
        all_data = search_table.to_pandas()
        print(f"Retrieved {len(all_data)} total documents from database")
        if len(all_data) == 0:
            print("No data in table")
            return "", None
            
        # Get embeddings and texts from database
        chunk_embs = np.array([row["vector"] for _, row in all_data.iterrows()])
        chunk_texts = [row["text"] for _, row in all_data.iterrows()]
        
        # Use custom cosine similarity search
        similarity_results = cosine_similarity_search(
            query, tokenizer, model, chunk_embs, chunk_texts, 
            top_k=num_results, device=device
        )
        
        print(f"✅ Found {len(similarity_results)} similar chunks for query: {query}")
        print(similarity_results)
        
        # Convert to a format compatible with the rest of the code
        # Create a dataframe with similar structure to what LanceDB would return
        results_data = []
        for i, (text, score) in enumerate(similarity_results):
            # Find the original row for this text to get metadata
            original_row = all_data[all_data["text"] == text].iloc[0] if any(all_data["text"] == text) else None
            if original_row is not None:
                # Add as a row with _distance instead of similarity (1-similarity for cosine distance)
                metadata = original_row["metadata"]
                # Convert any numpy types to native Python types in metadata
                sanitized_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, np.ndarray):
                        sanitized_metadata[k] = v.tolist()
                    elif isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
                        sanitized_metadata[k] = v.item()
                    else:
                        sanitized_metadata[k] = v
                
                results_data.append({
                    "text": text,
                    "_distance": 1.0 - score,  # Convert similarity to distance
                    "metadata": sanitized_metadata
                })
        
        results = pd.DataFrame(results_data)
        
        # Save similarity scores to a JSON file if we have results
        if len(results) > 0:
            print(f"Found {len(results)} results using vector search")
            
            # Create a directory for similarity scores if it doesn't exist
            os.makedirs("similarity_scores", exist_ok=True)
            
            # Prepare data for JSON
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            similarity_data = {
                "query": query,
                "timestamp": timestamp,
                "results": []
            }
            
            # Extract relevant data including similarity scores
            for idx, row in results.iterrows():
                metadata = row.get("metadata", {})
                
                # Convert any numpy types to native Python types
                page_numbers = metadata.get("page_numbers", [])
                if isinstance(page_numbers, np.ndarray):
                    page_numbers = page_numbers.tolist()
                elif not isinstance(page_numbers, list):
                    page_numbers = [page_numbers] if page_numbers else []
                
                # Build sanitized metadata dictionary
                sanitized_metadata = {
                    "context": str(metadata.get("context", "")),
                    "filename": str(metadata.get("filename", "")),
                    "page_numbers": page_numbers,
                    "title": str(metadata.get("title", ""))
                }
                
                similarity_data["results"].append({
                    "text": str(row.get("text", "")),
                    "similarity_score": float(1 - float(row["_distance"])),  # cosine similarity
                    "metadata": sanitized_metadata
                })
            
            try:
                # Save to file
                score_file = os.path.join("similarity_scores", f"query_{timestamp}.json")
                with open(score_file, "w") as f:
                    json.dump(similarity_data, f, indent=2)
                print(f"Similarity scores saved to {score_file}")
            except Exception as json_error:
                print(f"Error saving similarity scores: {json_error}")
                # Continue processing even if saving fails
            
            print(f"Similarity scores saved to {score_file}")
    except Exception as e:
        print(f"Vector search failed: {e}")
        # If vector search fails, log error and return empty results
        print("No fallback search available, using only vector search")
        st.warning("Search functionality is currently limited to vector search only.")
        return "", None

    # Only process results if we have any
    print(f"Building contexts from {len(results) if results is not None else 0} results...")
    if results is not None and len(results) > 0:
        for idx, row in results.iterrows():
            # Extract metadata safely
            try:
                # Debug print to check what's in the row
                print(f"Processing result {idx+1}: {len(row['text'])} characters of text")
                
                filename = row["metadata"].get("filename", "")
                page_numbers = row["metadata"].get("page_numbers", [])
                title = row["metadata"].get("title", "")

                # Build source citation
                source_parts = []
                if filename:
                    source_parts.append(filename)
                if page_numbers is not None:
                    # Ensure page_numbers is properly formatted regardless of type
                    if isinstance(page_numbers, (list, tuple, np.ndarray)):
                        page_str = ", ".join(str(p) for p in page_numbers)
                    else:
                        # Single value
                        page_str = str(page_numbers)
                    source_parts.append(f"p. {page_str}")

                source = f"\nSource: {' - '.join(source_parts)}"
                if title:
                    source += f"\nTitle: {title}"

                # Create context entry with the text and source info
                context_entry = f"{row['text']}{source}"
                contexts.append(context_entry)
                print(f"Added context {idx+1} of length {len(context_entry)}")
                
            except Exception as row_err:
                print(f"Error processing search result {idx}: {row_err}")
                # Add a simpler version without the metadata that's causing problems
                try:
                    simple_entry = f"{row.get('text', 'No text available')}\nSource: Unknown"
                    contexts.append(simple_entry)
                    print(f"Added simplified context {idx+1}")
                except Exception as e:
                    print(f"Failed to add even simplified context: {e}")
    
    # Print debug info about the contexts
    print(f"Total number of contexts built: {len(contexts)}")
    for i, ctx in enumerate(contexts):
        print(f"Context {i+1} length: {len(ctx)} characters")
    
    if not contexts:
        # If we still have no contexts, provide a fallback message
        print("No valid contexts found in search results")
        context_text = "No relevant information found in the document. Please try rephrasing your question or upload a different document."
        return context_text, results
    else:
        # Join all the valid contexts
        context_text = "\n\n".join(contexts)
        print(f"Final context length: {len(context_text)} characters")
        print("----- Retrieved context -----")
        print(context_text)
        print("-----------------------------")
        return context_text, results


def generate_highlight_annotations(document, excerpts):
    """Generate highlight annotations for PDF excerpts.
    
    Args:
        document: PyMuPDF document object
        excerpts: List of text excerpts to highlight
        
    Returns:
        List of annotation dictionaries
    """
    annotations = []
    for page_num, page in enumerate(document):
        for excerpt in excerpts:
            for inst in page.search_for(excerpt):
                annotations.append({
                    "page": page_num + 1,
                    "x": inst.x0, "y": inst.y0,
                    "width": inst.x1 - inst.x0,
                    "height": inst.y1 - inst.y0,
                    "color": "red",
                })
    return annotations


def extract_excerpts(text, max_length=50, overlap=5):
    """Extract meaningful excerpts from longer text.
    
    Args:
        text: Text to extract excerpts from
        max_length: Maximum length of each excerpt
        overlap: Overlap between excerpts
        
    Returns:
        List of text excerpts
    """
    words = text.split()
    excerpts = []
    
    if len(words) <= max_length:
        return [text]
        
    for i in range(0, len(words), max_length - overlap):
        excerpt = ' '.join(words[i:i + max_length])
        if excerpt:
            excerpts.append(excerpt)
            
    return excerpts


def get_chat_response(messages, context: str) -> str:
    """Get streaming response from OpenAI API.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer questions. If you're unsure or the context
    doesn't contain the relevant information, say so.
    
    Context:
    {context}
    """

    messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

    print("----- Retrieved context -----")
    print(context)
    print("-----------------------------")

    # Create the streaming response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_context,
        temperature=st.session_state.temperature,
        stream=True,
    )

    # Use Streamlit's built-in streaming capability
    response = st.write_stream(stream)
    return response


def display_pdf_with_highlights(filename, page_numbers, excerpts):
    """Display PDF with highlighted excerpts.
    
    Args:
        filename: PDF filename
        page_numbers: List of page numbers to show
        excerpts: List of text excerpts to highlight
    """
    pdf_path = os.path.join(PDF_DIR, filename)
    
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found: {pdf_path}")
        return
    
    # Open the PDF document
    try:
        document = fitz.open(pdf_path)
        
        # Generate highlight annotations
        annotations = generate_highlight_annotations(document, excerpts)
        
        # Create PDF viewer
        pdf_tabs = st.tabs([f"Page {p}" for p in page_numbers])
        
        for i, page_num in enumerate(page_numbers):
            with pdf_tabs[i]:
                # Display PDF page with highlights
                # Convert page_num to standard Python int to ensure JSON serialization
                page_num_int = int(page_num)
                
                # Only render the current page where the highlight is located
                pages_to_render = [page_num_int]
                
                # Filter annotations for only this page
                page_annotations = []
                for a in annotations:
                    if int(a["page"]) == page_num_int:
                        # Convert all numeric values to standard Python types
                        page_annotations.append({
                            "page": int(a["page"]),
                            "x": float(a["x"]),
                            "y": float(a["y"]),
                            "width": float(a["width"]),
                            "height": float(a["height"]),
                            "color": a["color"]
                        })
                
                # Use the correct parameters based on documentation
                pdf_viewer(
                    pdf_path, 
                    width=700,  # Default width from docs
                    annotations=page_annotations,
                    scroll_to_page=page_num_int,
                    render_text=True,
                    pages_to_render=pages_to_render  # Only render the current page
                )
                
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")


# Initialize Streamlit app
st.title("📚 Document Q&A")

# Check if API key exists, if not, show input form
if not check_api_key():
    st.warning("⚠️ OpenAI API key is missing. Please enter your API key to continue.")
    
    with st.form("api_key_form"):
        api_key = st.text_input("OpenAI API Key", type="password")
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            if api_key and len(api_key) > 20:  # Simple validation for key format
                if save_api_key(api_key):
                    st.success("✅ API key saved successfully! Initializing application...")
                    # Initialize the OpenAI client with the new key
                    client = OpenAI()
                    # Rerun the app to refresh with the new API key
                    st.rerun()
                else:
                    st.error("Failed to save API key. Please check file permissions.")
            else:
                st.error("Invalid API key format. Please enter a valid OpenAI API key.")
    
    # Stop the app here if no API key is provided
    st.stop()

# Initialize session state for processed files if it doesn't exist
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Sidebar for file upload
st.sidebar.header("Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "txt"], key="file_upload")

# Sidebar for model settings
st.sidebar.header("Model Settings")
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7  # Default value

temperature = st.sidebar.slider(
    "Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=st.session_state.temperature, 
    step=0.1,
    help="Lower values make responses more deterministic, higher values more creative"
)
st.session_state.temperature = temperature

# Add to your sidebar settings section
st.sidebar.header("Search Settings")
num_results = st.sidebar.slider(
    "Number of chunks to retrieve", 
    min_value=1, 
    max_value=10, 
    value=3, 
    step=1,
    help="Higher values retrieve more context but may include less relevant information"
)
 
if uploaded_file:
    # Check if this file has already been processed
    file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if file_identifier not in st.session_state.processed_files:
        # Hide the file from the UI before processing
        with st.sidebar.status("Processing document..."):
            # Determine the right target path based on file type
            if uploaded_file.name.lower().endswith('.pdf'):
                # For PDFs, save directly to PDF_DIR
                saved_filepath = os.path.join(PDF_DIR, uploaded_file.name)
            else:
                # For other files, save to a general uploads directory
                uploads_dir = "data/uploads"
                os.makedirs(uploads_dir, exist_ok=True)
                saved_filepath = os.path.join(uploads_dir, uploaded_file.name)
            
            # Write the file to disk
            with open(saved_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Process document using the saved file path
            status = process_document(saved_filepath, uploaded_file.name)
            
            # Add file to processed files
            st.session_state.processed_files.add(file_identifier)
            
            st.sidebar.success(f"{status} (Saved at: {saved_filepath})")

            st.rerun()
    else:
        st.sidebar.info(f"'{uploaded_file.name}' has already been processed.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for PDF display
if "pdf_info" not in st.session_state:
    st.session_state.pdf_info = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    st.header("🔍 Search Results: Relevant Sections")
    with st.status("View Relevant Sections", expanded=False) as status:
        context, results = get_context(prompt, table, num_results=num_results)
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.write("Found relevant sections:")
        
        # Store PDF information for each result
        pdf_info = {}
        if results is None or (hasattr(results, 'empty') and results.empty):
            st.warning("No results found for your query. Try rephrasing it.")
        else:
            for _, row in results.iterrows():
                # Extract text and metadata
                text = row["text"]
                filename = row["metadata"]["filename"]
                page_numbers = row["metadata"]["page_numbers"]
                title = row["metadata"]["title"]
                
                # Extract excerpts for highlighting
                excerpts = extract_excerpts(text)
                
                # Add to PDF info for later display
                if filename.endswith('.pdf'):
                    if filename not in pdf_info:
                        pdf_info[filename] = {
                            "page_numbers": set(),
                            "excerpts": set()
                        }
                    pdf_info[filename]["page_numbers"].update(page_numbers)
                    pdf_info[filename]["excerpts"].update(excerpts)
                
                # Build display information
                source = f"{filename}"
                if page_numbers is not None and len(page_numbers) > 0:
                    source += f" - p. {', '.join(str(p) for p in page_numbers)}"
                    
                st.markdown(
                    f"""
                    <div class="search-result">
                        <details>
                            <summary>{source}</summary>
                            <div class="metadata">Section: {title}</div>
                            <div style="margin-top: 8px;">{text}</div>
                        </details>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
        
        # Save PDF info to session state
        st.session_state.pdf_info = pdf_info

    # Display assistant response
    with st.chat_message("assistant"):
        # Get model response with streaming
        response = get_chat_response(st.session_state.messages, context)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display PDFs with highlights
if st.session_state.pdf_info:
    st.header("📄 Document Highlights")
    
    for filename, info in st.session_state.pdf_info.items():
        with st.expander(f"View {filename}"):
            # Convert to standard Python types
            page_numbers = [int(p) for p in sorted(list(info["page_numbers"]))]
            excerpts = [str(e) for e in list(info["excerpts"])]
            
            display_pdf_with_highlights(
                filename=filename,
                page_numbers=page_numbers,
                excerpts=excerpts
            )