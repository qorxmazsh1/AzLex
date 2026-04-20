import os
import json
import re
import glob

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import subprocess
    ANTIWORD_AVAILABLE = True
except:
    ANTIWORD_AVAILABLE = False


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def read_doc_file(filepath):

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.docx' and DOCX_AVAILABLE:
        try:
            doc = DocxDocument(filepath)
            text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            if len(text) > 100:
                return text
        except Exception as e:
            print(f"    python-docx failed: {e}")

    if ext == '.doc' and DOCX_AVAILABLE:
        try:
            doc = DocxDocument(filepath)
            text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            if len(text) > 100:
                return text
        except Exception as e:
            print(f"    .doc direct read failed: {e}")

    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        text = raw.decode('utf-8', errors='ignore')
        text = re.sub(r'[^\x20-\x7E\u00C0-\u024F\u0400-\u04FF]', ' ', text)
        text = clean_text(text)
        if len(text) > 500:
            return text
    except Exception as e:
        print(f"    Raw read failed: {e}")

    return None


def chunk_text(text, chunk_size=1000, overlap=200):

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            boundary = text.rfind('.', start + chunk_size - 200, end)
            if boundary != -1:
                end = boundary + 1

        chunks.append(text[start:end].strip())
        start += chunk_size - overlap

    return [c for c in chunks if len(c) > 50]  


def get_law_name(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.lstrip('-_')
    return name.strip()


def process_docs_folder(docs_folder="Docs", output_dir="law_data"):
   
    os.makedirs(output_dir, exist_ok=True)

    
    doc_files = (
        glob.glob(os.path.join(docs_folder, "*.doc")) +
        glob.glob(os.path.join(docs_folder, "*.docx"))
    )

    if not doc_files:
        print(f"No .doc/.docx files found in {docs_folder}/")
        return

    print(f"Found {len(doc_files)} files in {docs_folder}/\n")

    all_laws = []
    all_chunks = []

    for filepath in doc_files:
        law_name = get_law_name(filepath)
        print(f"Processing: {law_name}")

        text = read_doc_file(filepath)

        if not text:
            print(f"  [!] Could not extract text from {filepath}\n")
            continue

        text = clean_text(text)
        print(f"  ok: {len(text):,} characters extracted")

        law_entry = {
            "name": law_name,
            "file": os.path.basename(filepath),
            "text": text,
            "char_count": len(text)
        }
        all_laws.append(law_entry)

        chunks = chunk_text(text)
        print(f"  chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id":       f"{law_name}_chunk_{i}",
                "law_name": law_name,
                "file":     os.path.basename(filepath),
                "chunk_id": i,
                "text":     chunk
            })

        print()


    raw_path = os.path.join(output_dir, "laws_raw.json")
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=2)

    chunks_path = os.path.join(output_dir, "laws_chunked.json")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Laws processed:  {len(all_laws)}")
    print(f"Total chunks:    {len(all_chunks)}")
    print(f"Raw JSON:        {raw_path}")
    print(f"Chunked JSON:    {chunks_path}")


if __name__ == "__main__":
    process_docs_folder(docs_folder="Docs", output_dir="law_data")