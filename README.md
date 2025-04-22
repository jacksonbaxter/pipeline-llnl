# Building Knowledge Extraction Pipeline with Docling

[Docling](https://github.com/DS4SD/docling) is a powerful, flexible open source document processing library that converts various document formats into a unified format. It has advanced document understanding capabilities powered by state-of-the-art AI models for layout analysis and table structure recognition.

## Key Features

- **Universal Format Support**: Process PDF, DOCX, XLSX, PPTX, Markdown, HTML, images, and more
- **Advanced Understanding**: AI-powered layout analysis and table structure recognition
- **Flexible Output**: Export to HTML, Markdown, JSON, or plain text
- **High Performance**: Efficient processing on local hardware

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

### 4. Running the Pipeline
Run each step in order:

1. Extract document content:
   ```bash
   python 1-extraction.py
   ```
2. Chunk the documents:
   ```bash
   python 2-chunking.py
   ```
3. Generate embeddings and store in LanceDB:
   ```bash
   python 3-embedding.py
   ```
4. (Optional) Test search:
   ```bash
   python 4-search.py
   ```
5. Launch the Streamlit chat interface:
   ```bash
   streamlit run 5-chat.py
   ```
   Then open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Switching Local Models (e.g., Qwen, Llama, etc.)
- Qwen is used as a local model by default. To use another local model (such as Llama or a different Qwen variant):
  1. Download your desired model and place it in an accessible location.
  2. Change the `QWEN_MODEL_NAME` variable in your `.env` file to the new model's name or path.
  3. (Re)run the pipeline scripts as needed.
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
| PDF | Native PDF documents with layout preservation |
| DOCX, XLSX, PPTX | Microsoft Office formats (2007+) |
| Markdown | Plain text with markup |
| HTML/XHTML | Web documents |
| Images | PNG, JPEG, TIFF, BMP |
| USPTO XML | Patent documents |
| PMC XML | PubMed Central articles |

Check out this [page](https://ds4sd.github.io/docling/supported_formats/) for an up to date list.

### Processing Pipeline

The standard pipeline includes:

1. Document parsing with format-specific backend
2. Layout analysis using AI models
3. Table structure recognition
4. Metadata extraction
5. Content organization and structuring
6. Export formatting