import streamlit as st
st._config.set_option("runner.moduleExcludedFromWatching", ["torch", "torchvision", "torchaudio"])

import lancedb
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os
import shutil
from streamlit_pdf_viewer import pdf_viewer
import tempfile
from extraction import extract_document
from chunking import chunk_document
from embedding import embed_document, Chunks

# Load environment variables
load_dotenv()

# Initialize OpenAI client
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
    
# Initialize database connection
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
        temperature=0.7,
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

# Initialize session state for processed files if it doesn't exist
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Sidebar for file upload
st.sidebar.header("Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "txt"], key="file_upload")

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
    with st.status("Searching document...", expanded=False) as status:
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

# Sidebar section to show existing PDFs in the data/pdfs directory
st.sidebar.subheader("📂 Stored PDFs")

pdf_files = [f for f in os.listdir("data/uploads")]

# Add a search bar
search_query_documents = st.sidebar.text_input("🔍 Search PDFs", "")

# Filter PDFs based on search query
filtered_pdfs = [pdf for pdf in sorted(pdf_files) if search_query_documents.lower() in pdf.lower()]

with st.sidebar.expander("View stored PDFs"):
    if filtered_pdfs:
        for pdf in filtered_pdfs:
            st.write(f"- {pdf}")
    else:
        st.write("No matching PDFs found.")