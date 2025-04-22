# Docling-Powered Document Q&A System

[Docling](https://github.com/DS4SD/docling) is a powerful, flexible open source document processing library that converts various document formats into a unified format. It has advanced document understanding capabilities powered by state-of-the-art AI models for layout analysis and table structure recognition.

## Key Features

- **Hybrid chunking using Docling**: Documents are split into semantically meaningful chunks for better retrieval and context.
- **LanceDB vector search**: Chunks are embedded and stored in LanceDB for fast, semantic search.
- **LLM summarization of results**: Retrieved chunks are summarized or synthesized by an LLM to answer user queries.
- **PDF highlighting**: Chunks relevant to the answer are highlighted directly within the PDF viewer interface.

## Streamlit UI Features

- **Embedding Method Selection**: Choose between Contextual Qwen (local model) and OpenAI Simple for generating embeddings.
- **Document Management**: Upload PDF, DOCX, or TXT files (up to 200MB each) via drag-and-drop or file browser. Manage documents for each embedding method, including deleting documents from the database.
- **Model Settings**: Adjust the temperature of the LLM to control the creativity of responses.
- **Search Settings**: Select the number of chunks to retrieve per query, allowing you to control the amount of context provided in answers.

## Step-by-Step Installation & Usage Guide

### 1. Prerequisites
- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- (Optional) [virtualenv](https://virtualenv.pypa.io/en/latest/) for isolated environments

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables & API Keys
Create a `.env` file in the project root with the following content:

```env
OPENAI_API_KEY=your_openai_api_key_here
# Name of the local model to use (e.g., Qwen, Llama, etc.)
QWEN_MODEL_NAME=Alibaba-NLP/gte-Qwen2-1.5B-instruct
```
- Only the OpenAI key and (optionally) a key for your local model are required.
- If using a different local model, set `QWEN_MODEL_NAME` to the name or path of your downloaded model.
- If your local model requires a key, add it to `.env` as needed (e.g., `LOCAL_MODEL_KEY=your_key`).
- The variable `USE_CONTEXTUAL_EMBEDDINGS` is not used and can be ignored.

### 4. Running the Q&A System

To use the document Q&A system, you only need to launch the Streamlit chat interface:

```bash
streamlit run chat.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

All document processing, chunking, embedding, and search are handled automatically through the chat interface.

### 5. Embedding Modes: Simple vs. Contextual Retrieval

The system supports two modes for generating embeddings:

**1. Simple Embedding Mode**
- OpenAI is used for both generating the context and creating the embeddings.
- This is straightforward and works well for general use cases.

**2. Contextual Retrieval Mode**
- OpenAI is used to create the context for retrieval.
- A local embedding model (such as Qwen, Llama, etc.) is used to generate the embeddings for each chunk along with the context of that chunk within the document.
- Embedding the context along with each chunk leads to better retrieval results, as the model can capture richer semantic relationships within the document.

#### Switching Between Modes and Models
- By default, Qwen is used as the local embedding model if configured.
- To use a different local model, update the `QWEN_MODEL_NAME` variable in your `.env` file to the desired model's name or path.
- If you want to use only OpenAI for embeddings, leave the local model configuration blank or unset.
- No code changes are required—just update the `.env` file.
- Make sure your model is compatible with the pipeline's embedding or inference code.

### 6. Notes
- Only OpenAI and local model keys are needed in `.env`.
- For more advanced usage or troubleshooting, see the comments in each script.
- If you encounter errors with local models, verify the model path and compatibility.

## Document Processing

### Supported Input Formats

| Format | Description |
|--------|-------------|
| PDF    | Native PDF documents with layout preservation |
| DOCX   | Microsoft Word documents |
| TXT    | Plain text files |

Only PDF, DOCX, and TXT files are currently supported as input for document Q&A.