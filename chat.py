import streamlit as st
import lancedb
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os
import re
import difflib
import shutil
from streamlit_pdf_viewer import pdf_viewer
from extraction import extract_document
from chunking import chunk_document
from embedding import embed_document, Chunks

# Load environment variables
load_dotenv()

# Check if OpenAI API key exists
def check_api_key():
    """Check if OpenAI API key exists in environment variables."""
    return os.getenv("OPENAI_API_KEY") is not None

# Function to save API key to .env file
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

# Initialize OpenAI client (only if API key exists)
client = None
if check_api_key():
    client = OpenAI()

# Define PDF directory - update this to where your PDFs are stored
PDF_DIR = "data/pdfs"

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
    db = lancedb.connect("data/lancedb")
    try:
        # Try to open the existing table
        table = db.open_table("docling")
        return table
    except Exception:
        # Create a new table if it doesn't exist
        db.create_table("docling", schema=Chunks, mode="create")
        table = db.open_table("docling")
        return table

table = init_db()

def process_document(file_path, file_name):
    """Run extraction, chunking, and embedding using existing modules.
    For PDFs, also save a copy in the PDF_DIR if needed.
    """
    # Only copy the file if it's not already in the PDF_DIR
    pdf_dest_path = os.path.join(PDF_DIR, file_name)
    if file_path != pdf_dest_path and file_name.lower().endswith('.pdf'):
        shutil.copyfile(file_path, pdf_dest_path)
        print(f"✅ {file_name} saved to {PDF_DIR}")
    
    document = extract_document(file_path)
    print(f"✅ {file_name} extracted!")
    chunks = chunk_document(document)
    print(f"✅ {file_name} chunked!")
    embed_document(chunks, existing_table=table)
    print(f"✅ {file_name} embedded!")
    return f"✅ {file_name} processed and stored successfully."

def get_context(query: str, table, num_results: int = 3) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    results = table.search(query).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Extract metadata
        filename = row["metadata"]["filename"]
        page_numbers = row["metadata"]["page_numbers"]
        title = row["metadata"]["title"]

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers is not None and len(page_numbers) > 0:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts), results


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

    # Create the streaming response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_context,
        temperature=st.session_state.temperature,  # Use temperature from session state
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

# Add temperature control slider to sidebar
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

# Initialize database connection
table = init_db()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def find_close_filename(user_input, filenames, cutoff=0.6):
    """Try to match a user-given filename or keyword to actual uploaded docs."""
    possible_names = re.findall(r'[\w\-.]+(?:\.pdf|\.docx|\.txt)?', user_input.lower())
    
    # Try exact or partial filename match
    for name in possible_names:
        matches = difflib.get_close_matches(name, filenames, n=1, cutoff=cutoff)
        if matches:
            return matches[0]
    
    # Fallback: keyword-based fuzzy match (just look for closest file)
    matches = difflib.get_close_matches(user_input.lower(), filenames, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def is_delete_request(prompt, filenames):
    """Check if prompt asks to delete and return the closest filename."""
    if re.search(r"\b(delete|remove|discard|erase)\b", prompt, re.IGNORECASE):
        best_match = find_close_filename(prompt, filenames)
        if best_match:
            return True, best_match
    return False, None

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

def delete_file_from_storage(filename):
    """Safely delete a file from data/pdfs or data/uploads directory."""
    directories = ["data/pdfs", "data/uploads"]
    deleted = False

    for directory in directories:
        full_path = os.path.join(directory, filename)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                deleted = True
                print(f"🗑️ Deleted file: {full_path}")
        except Exception as e:
            print(f"❌ Failed to delete {full_path}: {e}")

    if not deleted:
        print(f"⚠️ File '{filename}' not found in pdfs or uploads.")
    return deleted

def list_uploaded_documents(table):
    """Return a list of all unique filenames in the LanceDB table."""
    df = table.to_pandas()
    return sorted(df['metadata'].apply(lambda x: x['filename']).dropna().unique())

uploaded_docs = list_uploaded_documents(table)

def delete_document_by_filename(table, filename):
    """Delete all rows in the table associated with a given filename."""
    condition = f'metadata["filename"] == "{filename}"'
    table.delete(condition)
    return f"✅ Document '{filename}' and its embeddings have been removed."

st.sidebar.header("🗂️ Manage Documents")

search_query = st.sidebar.text_input("Search for a document (optional)")

uploaded_docs = list_uploaded_documents(table)

filtered_docs = (
    [doc for doc in uploaded_docs if search_query.lower() in doc.lower()]
    if search_query
    else uploaded_docs
)

st.sidebar.caption(f"📄 {len(filtered_docs)} document(s) shown out of {len(uploaded_docs)} total.")

filename_to_delete = st.sidebar.selectbox("Select a document to delete", filtered_docs)

if st.sidebar.button("Delete Selected Document"):
    msg1 = delete_document_by_filename(table, filename_to_delete)
    deleted = delete_file_from_storage(filename_to_delete)
    msg2 = f"🗑️ File '{filename_to_delete}' deleted from storage." if deleted else f"⚠️ File '{filename_to_delete}' not found on disk."
    st.sidebar.success(f"{msg1}\n{msg2}")
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the current uploaded documents
    uploaded_docs = list_uploaded_documents(table)

    # Check for delete intent with fuzzy match
    is_delete, filename = is_delete_request(prompt, uploaded_docs)

    if is_delete:
        msg1 = delete_document_by_filename(table, filename)
        deleted = delete_file_from_storage(filename)
        msg2 = f"🗑️ File '{filename}' deleted from storage." if deleted else f"⚠️ File '{filename}' not found on disk."
        response = f"{msg1}\n{msg2}"

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

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