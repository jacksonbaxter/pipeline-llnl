import streamlit as st
import lancedb
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os
import shutil
from streamlit_pdf_viewer import pdf_viewer
from extraction import extract_document
from chunking import chunk_document
from embedding import embed_document, Chunks
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Check if API keys exist
def check_api_key(api_type):
    """Check if API key exists in environment variables."""
    if api_type == "openai":
        return os.getenv("OPENAI_API_KEY") is not None
    elif api_type == "google":
        return os.getenv("GEMINI_API_KEY") is not None
    return False

# Function to save API key to .env file
def save_api_key(api_key, api_type="openai"):
    """Save API key to .env file."""
    env_path = ".env"
    
    # Read existing .env content
    env_content = ""
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_content = f.read()
    
    # Set the appropriate environment variable name based on the API type
    env_var_name = "OPENAI_API_KEY" if api_type == "openai" else "GEMINI_API_KEY"
    
    # Check if the API key already exists in the file
    if f"{env_var_name}=" in env_content:
        # Replace existing key
        lines = env_content.splitlines()
        updated_lines = []
        for line in lines:
            if line.startswith(f"{env_var_name}="):
                updated_lines.append(f"{env_var_name}={api_key}")
            else:
                updated_lines.append(line)
        env_content = "\n".join(updated_lines)
    else:
        # Append new key
        if env_content and not env_content.endswith("\n"):
            env_content += "\n"
        env_content += f"{env_var_name}={api_key}"
    
    # Write back to .env file
    with open(env_path, "w") as f:
        f.write(env_content)
    
    # Update environment variable in current session
    os.environ[env_var_name] = api_key
    return True

# Initialize clients (only if API keys exist)
openai_client = None
if check_api_key("openai"):
    openai_client = OpenAI()

gemini_client = None
if check_api_key("google"):
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Model definitions
MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "display_name": "GPT-4o Mini (OpenAI)",
        "model_id": "gpt-4o-mini"
    },
    "gemini-2.0-flash": {
        "provider": "google",
        "display_name": "Gemini 2.0 Flash (Google)",
        "model_id": "gemini-2.0-flash"
    },
    "gemini-2.0-flash-lite": {
        "provider": "google",
        "display_name": "Gemini 2.0 Flash Lite (Google)",
        "model_id": "gemini-2.0-flash-lite"
    }
}

# Function to check if a model is available based on API key
def is_model_available(model_key):
    model_info = MODELS.get(model_key)
    if not model_info:
        return False
    
    if model_info["provider"] == "openai":
        return check_api_key("openai")
    elif model_info["provider"] == "google":
        return check_api_key("google")
    
    return False

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
    """Get streaming response from the selected model API.

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

    selected_model = st.session_state.get("model", "gpt-4o-mini")
    model_info = MODELS[selected_model]
    
    if model_info["provider"] == "openai":
        messages_with_context = [{"role": "system", "content": system_prompt}, *messages]
        
        # Create the streaming response
        stream = openai_client.chat.completions.create(
            model=model_info["model_id"],
            messages=messages_with_context,
            temperature=st.session_state.temperature,
            stream=True,
        )
        
        # Use Streamlit's built-in streaming capability
        response = st.write_stream(stream)
    
    elif model_info["provider"] == "google":
        # Format messages for Gemini API
        # Start with the system message
        formatted_content = [
            {"role": "user", "parts": [{"text": f"{system_prompt}\n\nFirst user message: {messages[0]['content']}"}]}
        ]
        
        # Add the rest of the conversation (starting from assistant's first response)
        for i in range(1, len(messages)):
            role = "model" if messages[i]["role"] == "assistant" else "user"
            formatted_content.append({"role": role, "parts": [{"text": messages[i]["content"]}]})
        
        # Create the streaming response
        stream = gemini_client.models.generate_content_stream(
            model=model_info["model_id"],
            contents=formatted_content,
            config=types.GenerateContentConfig(temperature=st.session_state.temperature),
        )

        # Adapt the Gemini stream for Streamlit's write_stream
        def gemini_stream_adapter():
            for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text

        # Use Streamlit's built-in streaming capability with the adapted stream
        response = st.write_stream(gemini_stream_adapter())
    
    else:
        response = "Error: Unsupported model provider"
    
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

# First, check if OpenAI API key exists, if not, show input form
if not check_api_key("openai"):
    st.warning("⚠️ OpenAI API key is required to continue.")
    
    with st.form("openai_api_key_form"):
        api_key = st.text_input("OpenAI API Key", type="password")
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            if api_key and len(api_key) > 20:  # Simple validation for key format
                if save_api_key(api_key, "openai"):
                    st.success("✅ OpenAI API key saved successfully! Initializing application...")
                    # Initialize the OpenAI client with the new key
                    openai_client = OpenAI()
                    # Rerun the app to refresh with the new API key
                    st.rerun()
                else:
                    st.error("Failed to save API key. Please check file permissions.")
            else:
                st.error("Invalid API key format. Please enter a valid OpenAI API key.")
    
    # Stop execution if OpenAI API key is not available
    st.stop()

# Initialize session state for processed files if it doesn't exist
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Sidebar for file upload
st.sidebar.header("Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "txt"], key="file_upload")

# Add model selector to sidebar
st.sidebar.header("Model Settings")
# Default to gpt-4o-mini or the first available model
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"

# Create model options with availability indicators
model_options = list(MODELS.keys())
model_display_names = []

for model_key in model_options:
    model_info = MODELS[model_key]
    available = is_model_available(model_key)
    status = "" if available else " (API Key Required)"
    model_display_names.append(f"{model_info['display_name']}{status}")

# Create a mapping of display names to model keys
display_name_to_key = {model_display_names[i]: model_options[i] for i in range(len(model_options))}

# Show model selector with visual indication of availability
selected_display_name = st.sidebar.selectbox(
    "Select Model",
    model_display_names,
    index=model_display_names.index(next(name for name, key in display_name_to_key.items() if key == st.session_state.model))
)

# Update the selected model in session state
previous_model = st.session_state.model
st.session_state.model = display_name_to_key[selected_display_name]

# Check if Google model is selected but no Google API key exists
selected_model_info = MODELS[st.session_state.model]
if selected_model_info["provider"] == "google" and not check_api_key("google"):
    st.warning(f"⚠️ Google API key is required to use {selected_model_info['display_name']}.")
    
    with st.form("google_api_key_form"):
        api_key = st.text_input("Google API Key", type="password")
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            if api_key and len(api_key) > 20:  # Simple validation for key format
                if save_api_key(api_key, "google"):
                    st.success("✅ Google API key saved successfully! Initializing application...")
                    # Initialize the Google client with the new key
                    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                    # Rerun the app to refresh with the new API key
                    st.rerun()
                else:
                    st.error("Failed to save API key. Please check file permissions.")
            else:
                st.error("Invalid API key format. Please enter a valid Google API key.")
    
    # Revert to previous model if Google API key is not provided
    st.session_state.model = previous_model
    st.rerun()

# Add temperature control slider to sidebar
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
        context, results = get_context(prompt, table)
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