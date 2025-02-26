import streamlit as st
import lancedb
from openai import OpenAI
from dotenv import load_dotenv
import os
from streamlit_pdf_viewer import pdf_viewer

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("data/lancedb")
    return db.open_table("docling")


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
        if page_numbers:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)


def extract_metadata_from_chunk(chunk):
    """Extract metadata from a chunk string.
    
    Args:
        chunk: String containing text and metadata
        
    Returns:
        tuple: (text, source_file, page_number, title)
    """
    # Split into text and metadata parts
    parts = chunk.split("\n")
    text = parts[0]
    
    # Initialize defaults
    source_file = None
    page_number = None
    title = "Untitled section"
    
    # Extract metadata
    for line in parts[1:]:
        if ": " in line:
            key, value = line.split(": ", 1)
            if key == "Source" and " - p. " in value:
                # Extract filename and page number
                file_parts = value.split(" - p. ")
                source_file = file_parts[0]
                page_number = int(file_parts[1]) if file_parts[1].isdigit() else None
            elif key == "Source":
                source_file = value
            elif key == "Title":
                title = value
                
    return text, source_file, page_number, title


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


# Initialize Streamlit app
st.title("📚 Document Q&A")

# Initialize session state for chat history and PDF viewer state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
    st.session_state.current_page = 0
    st.session_state.highlighted_text = None
    
# Track active tab
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # Default to Chat tab

# Create a function to handle chunk clicks
def handle_chunk_click(source_file, page_number, text):
    """Handle click on a document chunk by displaying the PDF and highlighting the text."""
    # PDF files are expected to be in a 'pdfs' directory
    pdf_path = os.path.join("pdfs", source_file)
    
    # Update session state
    st.session_state.current_pdf = pdf_path
    st.session_state.current_page = page_number if page_number else 1  # Use 1-indexed for scroll_to_page
    st.session_state.highlighted_text = text
    
    # Create annotation for the page - using the exact format from the documentation
    annotation = {
        "page": page_number if page_number else 1,  # Pages are 1-indexed in annotations
        "x": 100,           # Left position (adjust based on your PDFs)
        "y": 150,           # Top position (adjust based on your PDFs)
        "height": 100,      # Height of highlight area
        "width": 400,       # Width of highlight area
        "color": "rgba(255, 255, 0, 0.3)"  # Semi-transparent yellow
    }
    
    # Store the annotation in session state
    st.session_state.annotations = [annotation]  # Use a single annotation at a time

# Initialize database connection
table = init_db()

# Create tabs for Chat and Document Viewer
tab1, tab2 = st.tabs(["💬 Chat", "📄 Document"])

# Set the active tab based on session state
if st.session_state.active_tab == 1:
    # This is a workaround to activate the Document tab
    # We need to rerun the app to make the tab switch take effect
    ui_was_refreshed = st.session_state.get("ui_refreshed", False)
    if not ui_was_refreshed:
        st.session_state.ui_refreshed = True
        st.rerun()
        
# Reset the refresh flag when we're on the Chat tab
if st.session_state.active_tab == 0:
    st.session_state.ui_refreshed = False

# Chat tab
with tab1:
    # Chat container to hold messages and input
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input at the bottom
    if prompt := st.chat_input("Ask a question about the document"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
            
        # Get relevant context
        with st.status("Searching document...", expanded=False) as status:
            context = get_context(prompt, table)
            status.update(label="Search complete!", state="complete")
            
        # Display search results in a collapsible container
        with st.expander("View search results", expanded=True):
            for i, chunk in enumerate(context.split("\n\n")):
                text, source_file, page_number, title = extract_metadata_from_chunk(chunk)
                
                # Create a card-like container for each result
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**Source:** {source_file or 'Unknown'} {f'(p. {page_number})' if page_number else ''}")
                        st.markdown(f"*{title}*")
                        st.markdown(text)
                    
                    with col2:
                        # Add button to view in PDF
                        if source_file and source_file.lower().endswith('.pdf'):
                            if st.button(f"View PDF", key=f"view_{i}"):
                                handle_chunk_click(source_file, page_number, text)
                                # Set active tab index in session state
                                st.session_state.active_tab = 1  # Index 1 is the Document tab
                
                # Add separator between results
                if i < len(context.split("\n\n")) - 1:
                    st.divider()
            
        # Display assistant response
        with st.chat_message("assistant"):
            # Get model response with streaming
            response = get_chat_response(st.session_state.messages, context)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Document viewer tab
with tab2:
    if st.session_state.current_pdf:
        try:
            # Check if the file exists
            if os.path.exists(st.session_state.current_pdf):
                st.header(f"Document: {os.path.basename(st.session_state.current_pdf)}")
                
                # Show the text context above the PDF
                if st.session_state.highlighted_text:
                    with st.container():
                        st.markdown("### Referenced Text")
                        st.info(st.session_state.highlighted_text)
                        st.write("Look for this text on page", st.session_state.current_page + 1)
                
                # Create a container for the PDF viewer
                with st.container():
                    # Define annotations if they exist
                    annotations = st.session_state.annotations if "annotations" in st.session_state else None
                    
                    # Include controls for adjusting annotation position
                    with st.expander("Adjust Highlight Position", expanded=False):
                        if "annotations" in st.session_state and len(st.session_state.annotations) > 0:
                            col1, col2 = st.columns(2)
                            with col1:
                                x_pos = st.slider("X position", 0, 600, 
                                                int(st.session_state.annotations[0]["x"]),
                                                key="x_pos")
                                width = st.slider("Width", 50, 600, 
                                                int(st.session_state.annotations[0]["width"]),
                                                key="width")
                            with col2:
                                y_pos = st.slider("Y position", 0, 800, 
                                                int(st.session_state.annotations[0]["y"]),
                                                key="y_pos")
                                height = st.slider("Height", 50, 600, 
                                                int(st.session_state.annotations[0]["height"]),
                                                key="height")
                            
                            # Update annotation with new values
                            if st.button("Update Highlight"):
                                st.session_state.annotations[0]["x"] = x_pos
                                st.session_state.annotations[0]["y"] = y_pos
                                st.session_state.annotations[0]["width"] = width
                                st.session_state.annotations[0]["height"] = height
                                st.experimental_rerun()
                                
                            # Button to clear highlight
                            if st.button("Clear Highlight"):
                                st.session_state.annotations = []
                                st.experimental_rerun()
                    
                    # Display the PDF with the specific page and annotation highlighting
                    pdf_viewer(
                        st.session_state.current_pdf,
                        annotations=annotations,
                        scroll_to_page=st.session_state.current_page,
                        height=700,
                        width="100%",  # Ensure it fills the available space
                        render_text=True  # Enable text layer for copy-paste
                    )
            else:
                st.error(f"PDF file not found: {st.session_state.current_pdf}")
        except Exception as e:
            st.error(f"Error displaying PDF: {str(e)}")
    else:
        # Display a placeholder when no document is selected
        st.info("Select a document from the search results to view it here.")