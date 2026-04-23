"""
AzLex RAG Pipeline
Retrieves relevant law chunks from ChromaDB and generates
answers using Google Gemini API.
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

try:
    import chromadb

    CHROMADB_IMPORT_ERROR = None
except ImportError as exc:
    chromadb = None
    CHROMADB_IMPORT_ERROR = exc

try:
    from dotenv import load_dotenv

    DOTENV_IMPORT_ERROR = None
except ImportError as exc:
    DOTENV_IMPORT_ERROR = exc

    def load_dotenv():
        return None


try:
    from google import genai

    GENAI_IMPORT_ERROR = None
except ImportError as exc:
    genai = None
    GENAI_IMPORT_ERROR = exc

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_IMPORT_ERROR = None
except ImportError as exc:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION = "azerbaijani_laws"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"
TOP_K = 6
SEARCH_K = 10
LOW_CONFIDENCE_THRESHOLD = 0.24
MAX_CHUNKS_PER_FILE = 3
RULES_DIR = "rules"

SYSTEM_PROMPT = """Sən Azərbaycan hüquq sistemi üzrə ixtisaslaşmış AI assistentisən.
Sənin vəzifən istifadəçinin sualını yalnız təqdim edilən hüquqi mənbələr və əlavə qaydalar əsasında izah etməkdir.

Əsas prinsiplər:
- Qəti hökm, zəmanət, "mütləq bəraət", "cəza olmayacaq" kimi iddialar qurma.
- İstifadəçi hadisə danışırsa, əvvəlcə onun mümkün hüquqi təsnifatını göstər: cinayət, inzibati, mülki və ya intizam müstəvisi.
- Məsuliyyətin yaranması üçün hansı faktların və sübutların önəmli olduğunu izah et.
- İstifadəçinin xeyrinə ola biləcək müdafiə, istisnaedici hallar və yüngülləşdirici hallar varsa, bunu yalnız mənbələrdə dayaq olduqda qeyd et.
- "Ən yüngül mümkün hüquqi nəticə" ifadəsindən istifadə edə bilərsən, amma yalnız bunun qanuni əsasını izah etdikdə.
- İstifadəçini qanundan yayınmağa, sübut gizlətməyə, saxta sənəd hazırlamağa və ya yalan ifadə verməyə yönləndirmə.
- Cavab Azərbaycan dilində, aydın, praktik və mərhələli olsun.
- Cavabın hüquqi maarifləndirmə məqsədi daşıdığını qısa qeyd et; fərdi hüquqi məsləhət və vəkil əvəzi olmadığını bildir.
"""


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
    if genai is None:
        missing.append(f"google-genai ({GENAI_IMPORT_ERROR})")
    if SentenceTransformer is None:
        missing.append(
            f"sentence-transformers ({SENTENCE_TRANSFORMERS_IMPORT_ERROR})"
        )
    if DOTENV_IMPORT_ERROR is not None:
        missing.append(f"python-dotenv ({DOTENV_IMPORT_ERROR})")
    if missing:
        raise ImportError("Missing Python packages for RAG runtime: " + ", ".join(missing))


def humanize_rule_name(path):
    file_name = os.path.splitext(os.path.basename(path))[0]
    parts = file_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        file_name = parts[1]
    return file_name.replace("-", " ").replace("_", " ").strip().title()


def load_rules(rules_dir=RULES_DIR):
    if not os.path.isdir(rules_dir):
        return []

    rules = []
    for path in sorted(glob.glob(os.path.join(rules_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as file_obj:
            text = file_obj.read().strip()
        if text:
            rules.append({"name": humanize_rule_name(path), "text": text})

    return rules


def render_rules(rules):
    if not rules:
        return "Əlavə qayda faylı tapılmadı."

    rendered = []
    for index, rule in enumerate(rules, 1):
        rendered.append(f"[{index}] {rule['name']}\n{rule['text']}")
    return "\n\n".join(rendered)


def is_incident_question(question):
    lowered = question.lower()
    incident_keywords = [
        "başıma",
        "məni",
        "mənə",
        "polis",
        "saxladı",
        "şöbə",
        "ifadə",
        "izahat",
        "şikayət",
        "qəza",
        "vurmuşam",
        "vurub",
        "dava",
        "hədə",
        "təhdid",
        "oğurluq",
        "cinayət işi",
        "məsuliyyət",
        "məhkəmə",
    ]
    return any(keyword in lowered for keyword in incident_keywords)


class AzLexRAG:
    def __init__(self):
        ensure_runtime_dependencies()

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(MODEL_NAME)

        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION)
        print(f"  {self.collection.count():,} chunks loaded")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        self.gemini = genai.Client(api_key=api_key)

        self.rules = load_rules(RULES_DIR)
        self.rendered_rules = render_rules(self.rules)
        print(f"  Loaded {len(self.rules)} rule file(s)")
        print("  Gemini API ready\n")

    def retrieve(self, question, top_k=TOP_K):
        embedding = self.embed_model.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=max(top_k, SEARCH_K),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        seen_texts = set()
        per_file_counts = defaultdict(int)

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            file_name = metadata.get("file", "")
            if not document.strip() or document in seen_texts:
                continue
            if per_file_counts[file_name] >= MAX_CHUNKS_PER_FILE:
                continue

            chunk = {
                "text": document,
                "law_name": metadata["law_name"],
                "file": file_name,
                "chunk_id": metadata["chunk_id"],
                "article_refs": metadata.get("article_refs", ""),
                "score": round(1 - distance, 3),
            }

            chunks.append(chunk)
            seen_texts.add(document)
            per_file_counts[file_name] += 1

            if len(chunks) >= top_k:
                break

        return chunks

    def build_context(self, chunks):
        context_parts = []
        for index, chunk in enumerate(chunks, 1):
            article_refs = chunk["article_refs"] or "maddə nömrəsi aşkarlanmadı"
            context_parts.append(
                "\n".join(
                    [
                        f"--- Mənbə {index} ---",
                        f"Məcəllə: {chunk['law_name']}",
                        f"Maddələr: {article_refs}",
                        f"Uyğunluq: {chunk['score']}",
                        f"Mətn: {chunk['text']}",
                    ]
                )
            )
        return "\n\n".join(context_parts)

    def build_prompt(self, question, chunks):
        question_type = "Hadisə təsviri" if is_incident_question(question) else "Ümumi hüquqi sual"
        best_score = max((chunk["score"] for chunk in chunks), default=0.0)
        confidence_note = (
            "Axtarış nəticələrinin uyğunluğu zəif görünür. Yalnız dəstəklənən hissələri cavablandır."
            if best_score < LOW_CONFIDENCE_THRESHOLD
            else "Axtarış nəticələri istifadə oluna biləcək səviyyədədir."
        )

        context = self.build_context(chunks)

        return f"""{SYSTEM_PROMPT}

İstifadəçi sualının tipi: {question_type}
Axtarış qiymətləndirməsi: {confidence_note}

Əlavə qayda faylları:
{self.rendered_rules}

Mənbələr:
{context}

Cavabı bu strukturla yaz:
1. Qısa hüquqi qiymətləndirmə
2. Mümkün məsuliyyət və ya hüquqi nəticə
3. Müdafiə üçün vacib hallar
4. Qanuni olaraq indi atılmalı addımlar
5. İstinad edilən maddələr
6. Qeyd

Xüsusi tələblər:
- Əgər faktlar natamamdırsa, bunu açıq yaz və hansı faktların nəticəni dəyişəcəyini qeyd et.
- Əgər mənbələr sualı tam cavablandırmırsa, bunu açıq de.
- Maddə nömrələrini və məcəllə adlarını konkret yaz.
- İstifadəçi polis, istintaq, saxlanılma, ifadəvermə və ya məhkəmə riski barədə danışırsa, lisenziyalı vəkillə tez danışmağı qısa tövsiyə et.
- Mənbədə olmayan detal uydurma.

Sual: {question}

Cavab:
"""

    def generate(self, question, chunks):
        if not chunks:
            return (
                "Uyğun hüquqi mənbə tapılmadı. Sualı daha konkret maddə, sahə və ya hadisə "
                "detalları ilə yenidən yazın ki, qanuni əsaslarla cavab qurmaq mümkün olsun."
            )

        prompt = self.build_prompt(question, chunks)
        response = self.gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text

    def ask(self, question):
        print(f"Question: {question}\n")

        chunks = self.retrieve(question)
        print(f"Retrieved {len(chunks)} relevant chunks:")
        for chunk in chunks:
            article_refs = chunk["article_refs"] or "maddə qeyd olunmayıb"
            print(f"  [{chunk['score']}] {chunk['law_name']} | {article_refs}")

        print("\nGenerating answer...\n")
        answer = self.generate(question, chunks)
        sources = [
            {
                "law": chunk["law_name"],
                "score": chunk["score"],
                "articles": chunk["article_refs"] or "maddə qeyd olunmayıb",
            }
            for chunk in chunks
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


def interactive_mode(rag):
    print("AzLex — Azərbaycan Hüquq Assistenti")
    print("Çıxmaq üçün 'q' yazın")

    while True:
        question = input("Sualınız: ").strip()
        if question.lower() in ["q", "quit", "exit"]:
            print("Görüşənədək!")
            break
        if not question:
            continue

        result = rag.ask(question)
        print("\n" + "=" * 60)
        print("CAVAB:")
        print(result["answer"])
        print("\nMƏNBƏLƏR:")
        for source in result["sources"]:
            print(
                f"  • {source['law']} | {source['articles']} "
                f"(uyğunluq: {source['score']})"
            )
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AzLex RAG Pipeline")
    parser.add_argument("--question", "-q", type=str, help="Ask a single question")
    args = parser.parse_args()

    try:
        rag = AzLexRAG()
    except ImportError as exc:
        print(str(exc))
        return

    if args.question:
        result = rag.ask(args.question)
        print("\n" + "=" * 60)
        print("CAVAB:")
        print(result["answer"])
        print("\nMƏNBƏLƏR:")
        for source in result["sources"]:
            print(
                f"  • {source['law']} | {source['articles']} "
                f"(uyğunluq: {source['score']})"
            )
    else:
        interactive_mode(rag)


if __name__ == "__main__":
    main()
