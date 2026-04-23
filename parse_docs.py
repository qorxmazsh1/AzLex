import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from xml.etree import ElementTree

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


DEFAULT_DOCS_FOLDER = "Docs"
DEFAULT_OUTPUT_DIR = "law_data"
RAW_FILENAME = "laws_raw.json"
CHUNK_FILENAME = "laws_chunked.json"
MANIFEST_FILENAME = "ingestion_manifest.json"
MANIFEST_VERSION = 2
ANTIWORD_BINARY = shutil.which("antiword")


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except ValueError:
                continue


configure_console_output()


def normalize_text(text):
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_front_matter(text):
    match = re.search(r"\bMaddə\s+\d+(?:-\d+)?\.", text)
    if not match:
        return text

    front_matter = text[:match.start()]
    looks_like_index = "MÜNDƏRİCAT" in front_matter.upper()
    if looks_like_index or len(front_matter) > 1500:
        return text[match.start():].strip()

    return text


def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_ooxml_text(filepath):
    if not zipfile.is_zipfile(filepath):
        return None

    try:
        with zipfile.ZipFile(filepath) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        print(f"    OOXML read failed: {exc}")
        return None

    try:
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        root = ElementTree.fromstring(xml_bytes)
        paragraphs = []
        for paragraph in root.findall(".//w:p", namespace):
            texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
            combined = "".join(texts).strip()
            if combined:
                paragraphs.append(combined)
        text = "\n".join(paragraphs)
        return text if len(text) > 100 else None
    except ElementTree.ParseError as exc:
        print(f"    OOXML XML parse failed: {exc}")
        return None


def read_doc_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    text = read_ooxml_text(filepath)
    if text:
        return text

    if ext in {".docx", ".doc"} and DOCX_AVAILABLE:
        try:
            doc = DocxDocument(filepath)
            text = "\n".join(
                para.text for para in doc.paragraphs if para.text and para.text.strip()
            )
            if len(text) > 100:
                return text
        except Exception as exc:
            print(f"    python-docx failed: {exc}")

    if ext == ".doc" and ANTIWORD_BINARY:
        try:
            result = subprocess.run(
                [ANTIWORD_BINARY, filepath],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            text = result.stdout.strip()
            if len(text) > 100:
                return text
            if result.stderr.strip():
                print(f"    antiword warning: {result.stderr.strip()}")
        except Exception as exc:
            print(f"    antiword failed: {exc}")

    try:
        with open(filepath, "rb") as file_obj:
            raw = file_obj.read()
        text = raw.decode("utf-8", errors="ignore")
        text = re.sub(r"[^\x20-\x7E\u00C0-\u024F\u0400-\u04FF\u0100-\u017F]", " ", text)
        text = normalize_text(text)
        if len(text) > 500:
            return text
    except Exception as exc:
        print(f"    Raw read failed: {exc}")

    return None


def split_into_article_units(text):
    pattern = re.compile(r"(?=Maddə\s+\d+(?:-\d+)?\.)")
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return []

    units = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        unit = text[start:end].strip()
        if len(unit) > 80:
            units.append(unit)

    return units


def find_split_point(text, start, preferred_end):
    if preferred_end >= len(text):
        return len(text)

    search_start = max(start + 200, preferred_end - 400)
    for marker in ("\n\n", ". ", "; ", ": "):
        split_at = text.rfind(marker, search_start, preferred_end)
        if split_at != -1:
            return split_at + len(marker)

    return preferred_end


def split_large_chunk(text, chunk_size=1800, overlap=250):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        preferred_end = min(len(text), start + chunk_size)
        end = find_split_point(text, start, preferred_end)
        if end <= start:
            end = preferred_end

        chunk = text[start:end].strip()
        if len(chunk) > 80:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def chunk_text(text, chunk_size=1800, overlap=250):
    text = strip_front_matter(normalize_text(text))
    article_units = split_into_article_units(text)

    if not article_units:
        return split_large_chunk(text, chunk_size=chunk_size, overlap=overlap)

    chunks = []
    current_units = []
    current_length = 0

    for unit in article_units:
        if len(unit) > chunk_size:
            if current_units:
                chunks.append("\n\n".join(current_units))
                current_units = []
                current_length = 0
            chunks.extend(split_large_chunk(unit, chunk_size=chunk_size, overlap=overlap))
            continue

        projected = current_length + len(unit) + (2 if current_units else 0)
        if current_units and projected > chunk_size:
            chunks.append("\n\n".join(current_units))
            current_units = [unit]
            current_length = len(unit)
            continue

        current_units.append(unit)
        current_length = projected

    if current_units:
        chunks.append("\n\n".join(current_units))

    return [normalize_text(chunk) for chunk in chunks if len(chunk.strip()) > 80]


def get_law_name(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.lstrip("-_")
    return name.strip()


def slugify(value):
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    return slug or "law"


def extract_article_refs(text, limit=6):
    article_refs = re.findall(r"Maddə\s+\d+(?:-\d+)?", text)
    unique_refs = []
    for ref in article_refs:
        if ref not in unique_refs:
            unique_refs.append(ref)
        if len(unique_refs) >= limit:
            break
    return unique_refs


def make_chunk_id(file_name, doc_hash, chunk_index):
    source_key = slugify(os.path.splitext(file_name)[0])
    return f"{source_key}-{doc_hash[:12]}-{chunk_index:04d}"


def load_json_list(path):
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[!] Could not load {path}: {exc}")
        return []


def load_manifest(path):
    if not os.path.exists(path):
        return {"version": MANIFEST_VERSION, "docs": {}}

    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        if isinstance(data, dict) and isinstance(data.get("docs"), dict):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[!] Could not load manifest {path}: {exc}")

    return {"version": MANIFEST_VERSION, "docs": {}}


def build_law_entry(law_name, file_name, text, doc_hash):
    return {
        "name": law_name,
        "file": file_name,
        "doc_hash": doc_hash,
        "text": text,
        "char_count": len(text),
        "article_count": text.count("Maddə "),
    }


def build_chunk_entries(law_name, file_name, text, doc_hash):
    chunks = chunk_text(text)
    entries = []

    for chunk_index, chunk in enumerate(chunks):
        article_refs = extract_article_refs(chunk)
        entries.append(
            {
                "id": make_chunk_id(file_name, doc_hash, chunk_index),
                "law_name": law_name,
                "file": file_name,
                "doc_hash": doc_hash,
                "chunk_id": chunk_index,
                "article_refs": ", ".join(article_refs),
                "text": chunk,
            }
        )

    return entries


def process_docs_folder(
    docs_folder=DEFAULT_DOCS_FOLDER,
    output_dir=DEFAULT_OUTPUT_DIR,
    force_reprocess=False,
):
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, RAW_FILENAME)
    chunks_path = os.path.join(output_dir, CHUNK_FILENAME)
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)

    doc_files = sorted(
        glob.glob(os.path.join(docs_folder, "*.doc"))
        + glob.glob(os.path.join(docs_folder, "*.docx"))
    )

    if not doc_files:
        print(f"No .doc/.docx files found in {docs_folder}/")
        return {
            "new_files": 0,
            "updated_files": 0,
            "reused_files": 0,
            "removed_files": [],
            "failed_files": [],
            "total_laws": 0,
            "total_chunks": 0,
            "raw_path": raw_path,
            "chunks_path": chunks_path,
            "manifest_path": manifest_path,
        }

    existing_raw = load_json_list(raw_path)
    existing_chunks = load_json_list(chunks_path)
    existing_manifest = load_manifest(manifest_path)

    raw_by_file = {entry.get("file"): entry for entry in existing_raw if entry.get("file")}
    chunks_by_file = defaultdict(list)
    for chunk in existing_chunks:
        file_name = chunk.get("file")
        if file_name:
            chunks_by_file[file_name].append(chunk)

    current_files = {os.path.basename(path) for path in doc_files}
    removed_files = sorted(set(existing_manifest.get("docs", {})) - current_files)

    print(f"Found {len(doc_files)} files in {docs_folder}/\n")

    all_laws = []
    all_chunks = []
    manifest_docs = {}
    failed_files = []
    new_files = 0
    updated_files = 0
    reused_files = 0

    for filepath in doc_files:
        file_name = os.path.basename(filepath)
        law_name = get_law_name(filepath)
        file_hash = compute_file_hash(filepath)
        previous_doc = existing_manifest.get("docs", {}).get(file_name, {})
        can_reuse = (not force_reprocess) and (
            previous_doc.get("doc_hash") == file_hash
            and file_name in raw_by_file
            and bool(chunks_by_file[file_name])
        )

        if can_reuse:
            law_entry = raw_by_file[file_name]
            chunk_entries = sorted(
                chunks_by_file[file_name], key=lambda chunk: chunk.get("chunk_id", 0)
            )
            reused_files += 1
            print(f"Reusing:    {law_name} ({len(chunk_entries)} chunks)")
        else:
            print(f"Processing: {law_name}")
            text = read_doc_file(filepath)

            if not text:
                print(f"  [!] Could not extract text from {filepath}")
                if previous_doc and file_name in raw_by_file and chunks_by_file[file_name]:
                    law_entry = raw_by_file[file_name]
                    chunk_entries = sorted(
                        chunks_by_file[file_name], key=lambda chunk: chunk.get("chunk_id", 0)
                    )
                    print("  [i] Keeping previous parsed version for this file.\n")
                    failed_files.append(file_name)
                else:
                    print()
                    failed_files.append(file_name)
                    continue
            else:
                text = strip_front_matter(normalize_text(text))
                law_entry = build_law_entry(law_name, file_name, text, file_hash)
                chunk_entries = build_chunk_entries(law_name, file_name, text, file_hash)

                if previous_doc:
                    updated_files += 1
                    status_label = "updated"
                else:
                    new_files += 1
                    status_label = "new"

                print(f"  {status_label}: {len(text):,} characters extracted")
                print(f"  chunks: {len(chunk_entries)}\n")

        all_laws.append(law_entry)
        all_chunks.extend(chunk_entries)
        manifest_docs[file_name] = {
            "file": file_name,
            "law_name": law_entry.get("name", law_name),
            "doc_hash": law_entry.get("doc_hash", previous_doc.get("doc_hash", file_hash)),
            "char_count": law_entry.get("char_count", 0),
            "chunk_count": len(chunk_entries),
            "chunk_ids": [chunk["id"] for chunk in chunk_entries],
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

    with open(raw_path, "w", encoding="utf-8") as file_obj:
        json.dump(all_laws, file_obj, ensure_ascii=False, indent=2)

    with open(chunks_path, "w", encoding="utf-8") as file_obj:
        json.dump(all_chunks, file_obj, ensure_ascii=False, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "version": MANIFEST_VERSION,
                "docs": manifest_docs,
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print("Sync summary:")
    print(f"  New files:      {new_files}")
    print(f"  Updated files:  {updated_files}")
    print(f"  Reused files:   {reused_files}")
    print(f"  Removed files:  {len(removed_files)}")
    print(f"  Failed files:   {len(failed_files)}")
    print(f"  Laws written:   {len(all_laws)}")
    print(f"  Total chunks:   {len(all_chunks)}")
    print(f"  Raw JSON:       {raw_path}")
    print(f"  Chunked JSON:   {chunks_path}")
    print(f"  Manifest:       {manifest_path}")

    return {
        "new_files": new_files,
        "updated_files": updated_files,
        "reused_files": reused_files,
        "removed_files": removed_files,
        "failed_files": failed_files,
        "total_laws": len(all_laws),
        "total_chunks": len(all_chunks),
        "raw_path": raw_path,
        "chunks_path": chunks_path,
        "manifest_path": manifest_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parse law documents into incremental raw/chunked JSON files."
    )
    parser.add_argument("--docs-folder", default=DEFAULT_DOCS_FOLDER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-reprocess", action="store_true")
    args = parser.parse_args()

    process_docs_folder(
        docs_folder=args.docs_folder,
        output_dir=args.output_dir,
        force_reprocess=args.force_reprocess,
    )


if __name__ == "__main__":
    main()
