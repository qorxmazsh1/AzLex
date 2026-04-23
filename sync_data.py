import argparse
import os
import sys

from embed import load_chunks, sync_vector_db
from parse_docs import CHUNK_FILENAME, process_docs_folder


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except ValueError:
                continue


configure_console_output()


def main():
    parser = argparse.ArgumentParser(
        description="Run incremental parsing and embedding in one command."
    )
    parser.add_argument("--docs-folder", default="Docs")
    parser.add_argument("--output-dir", default="law_data")
    parser.add_argument("--force-reprocess", action="store_true")
    args = parser.parse_args()

    parse_summary = process_docs_folder(
        docs_folder=args.docs_folder,
        output_dir=args.output_dir,
        force_reprocess=args.force_reprocess,
    )

    chunks_path = os.path.join(args.output_dir, CHUNK_FILENAME)
    if not os.path.exists(chunks_path):
        print(f"Chunk file not found after parsing: {chunks_path}")
        return

    chunks = load_chunks(chunks_path)
    try:
        _, embed_summary = sync_vector_db(chunks)
    except ImportError as exc:
        print("\nEmbedding sync skipped:")
        print(f"  {exc}")
        print("  Parse mərhələsi uğurla tamamlandı.")
        return

    print("\nPipeline sync complete.")
    print(
        f"  Parse: new={parse_summary['new_files']}, "
        f"updated={parse_summary['updated_files']}, reused={parse_summary['reused_files']}"
    )
    print(
        f"  Embeddings: added={embed_summary['added']}, "
        f"deleted={embed_summary['deleted']}, total={embed_summary['total']}"
    )


if __name__ == "__main__":
    main()
