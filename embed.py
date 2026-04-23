import argparse
import json
import os
import sys

try:
    import chromadb

    CHROMADB_IMPORT_ERROR = None
except ImportError as exc:
    chromadb = None
    CHROMADB_IMPORT_ERROR = exc

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_IMPORT_ERROR = None
except ImportError as exc:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

CHUNKS_PATH = "law_data/laws_chunked.json"
CHROMA_DIR = "chroma_db"
COLLECTION = "azerbaijani_laws"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32
DELETE_BATCH_SIZE = 500


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except ValueError:
                continue


configure_console_output()


def ensure_runtime_dependencies():
    missing = []
    if chromadb is None:
        missing.append(f"chromadb ({CHROMADB_IMPORT_ERROR})")
    if SentenceTransformer is None:
        missing.append(
            f"sentence-transformers ({SENTENCE_TRANSFORMERS_IMPORT_ERROR})"
        )
    if missing:
        raise ImportError(
            "Missing Python packages for embedding sync: " + ", ".join(missing)
        )


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        chunks = json.load(file_obj)
    print(f"Loaded {len(chunks):,} chunks from {path}")
    return chunks


def batched(items, batch_size):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def sync_vector_db(chunks):
    ensure_runtime_dependencies()

    print(f"Setting up ChromaDB at: {CHROMA_DIR}/")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    existing_data = collection.get(limit=existing_count) if existing_count else {"ids": []}
    existing_ids = set(existing_data.get("ids", []))
    desired_ids = {chunk["id"] for chunk in chunks}
    stale_ids = sorted(existing_ids - desired_ids)

    if stale_ids:
        print(f"Removing {len(stale_ids):,} stale chunks from vector DB...")
        for batch in batched(stale_ids, DELETE_BATCH_SIZE):
            collection.delete(ids=batch)

    new_chunks = [chunk for chunk in chunks if chunk["id"] not in existing_ids]
    if not new_chunks:
        total_after_sync = collection.count()
        print("No new chunks to embed.")
        return collection, {
            "added": 0,
            "deleted": len(stale_ids),
            "total": total_after_sync,
        }

    print(MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    print(
        f"Embedding {len(new_chunks):,} new chunks "
        f"(deleting {len(stale_ids):,} stale chunks first)..."
    )

    for batch in tqdm(list(batched(new_chunks, BATCH_SIZE)), desc="Embedding"):
        texts = [chunk["text"] for chunk in batch]
        ids = [chunk["id"] for chunk in batch]
        metadatas = [
            {
                "law_name": chunk["law_name"],
                "chunk_id": chunk["chunk_id"],
                "file": chunk.get("file", ""),
                "doc_hash": chunk.get("doc_hash", ""),
                "article_refs": chunk.get("article_refs", ""),
            }
            for chunk in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    total_after_sync = collection.count()
    return collection, {
        "added": len(new_chunks),
        "deleted": len(stale_ids),
        "total": total_after_sync,
    }


def test_search(collection):
    ensure_runtime_dependencies()

    model = SentenceTransformer(MODEL_NAME)

    test_queries = [
        "işçinin əmək hüquqları",
        "cinayət məsuliyyəti nə vaxt yaranır",
        "yol hərəkəti qaydalarının pozulması",
    ]

    for query in test_queries:
        embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )

        print(f"\nQuery: '{query}'")
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            article_refs = metadata.get("article_refs") or "maddə qeyd olunmayıb"
            print(f"  [{metadata['law_name']}] (score: {1 - distance:.3f})")
            print(f"  Maddələr: {article_refs}")
            print(f"  {document[:120]}...")


def main():
    parser = argparse.ArgumentParser(description="Sync chunk embeddings into ChromaDB.")
    parser.add_argument("--chunks-path", default=CHUNKS_PATH)
    parser.add_argument("--skip-test-search", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.chunks_path):
        print(f"Error: {args.chunks_path} not found.")
        return

    chunks = load_chunks(args.chunks_path)
    try:
        collection, summary = sync_vector_db(chunks)
    except ImportError as exc:
        print(str(exc))
        return

    print(
        "\nVector sync summary: "
        f"added={summary['added']}, deleted={summary['deleted']}, total={summary['total']}"
    )

    if not args.skip_test_search:
        test_search(collection)

    print(f"\nVector DB ready at: {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
