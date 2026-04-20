import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH  = "law_data/laws_chunked.json"
CHROMA_DIR   = "chroma_db"
COLLECTION   = "azerbaijani_laws"

MODEL_NAME   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE   = 32


def load_chunks(path):
    with open(path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks):,} chunks from {path}")
    return chunks


def build_vector_db(chunks):
    print(MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print(f"Setting up ChromaDB at: {CHROMA_DIR}/")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}  
    )

    existing = set(collection.get()['ids'])
    new_chunks = [c for c in chunks if c['id'] not in existing]

    if not new_chunks:
        return collection

    print(f"Embedding {len(new_chunks):,} new chunks (skipping {len(existing):,} existing)...")

    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Embedding"):
        batch = new_chunks[i:i + BATCH_SIZE]

        texts = [c['text'] for c in batch]
        ids   = [c['id']   for c in batch]
        metas = [{"law_name": c['law_name'], "chunk_id": c['chunk_id'], "file": c.get('file', '')} for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metas
        )

    return collection


def test_search(collection):
    model = SentenceTransformer(MODEL_NAME)

    test_queries = [
        "işcinin emek huquqlari",
        "cinayet cezasi",
        "vergi odeme qaydalari",
    ]

    for query in test_queries:
        embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings = embedding,
            n_results        = 2,
            include          = ["documents", "metadatas", "distances"]
        )
        print(f"\nQuery: '{query}'")
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            print(f"  [{meta['law_name']}] (score: {1-dist:.3f})")
            print(f"  {doc[:120]}...")


def main():
    if not os.path.exists(CHUNKS_PATH):
        print(f"Error: {CHUNKS_PATH} not found.")
        return

    chunks = load_chunks(CHUNKS_PATH)

    collection = build_vector_db(chunks)

    test_search(collection)

    print(f"\nVector DB ready at: {CHROMA_DIR}/")


if __name__ == "__main__":
    main()