import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from typing import List

db = lancedb.connect("data/lancedb")

func = get_registry().get("openai").create(name="text-embedding-3-large")

class ChunkMetadata(LanceModel):
    filename: str | None
    page_numbers: List[int] | None
    title: str | None

class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()
    metadata: ChunkMetadata

def embed_document(chunks):
    """Embeds chunks into LanceDB."""
    table = db.create_table("docling", schema=Chunks, mode="overwrite")

    processed_chunks = [
        {
            "text": chunk.text,
            "metadata": {
                "filename": chunk.meta.origin.filename,
                "page_numbers": [
                    page_no
                    for page_no in sorted(
                        set(
                            prov.page_no
                            for item in chunk.meta.doc_items
                            for prov in item.prov
                        )
                    )
                ]
                or None,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            },
        }
        for chunk in chunks
    ]

    table.add(processed_chunks)
