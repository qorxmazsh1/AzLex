"""
AzLex RAG Pipeline
Retrieves relevant law chunks from ChromaDB and generates
answers using Google Gemini API.
"""

import os
import argparse
import chromadb
from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
CHROMA_DIR   = "chroma_db"
COLLECTION   = "azerbaijani_laws"
MODEL_NAME   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"
TOP_K        = 5

SYSTEM_PROMPT = """Sən Azərbaycan hüquq sistemi üzrə ixtisaslaşmış AI assistentisən.
Sənə verilən hüquqi maddələrə əsaslanaraq istifadəçinin sualına Azərbaycan dilində cavab ver.

Qaydalar:
1. Yalnız verilən maddələrdəki məlumata əsaslan
2. Hər cavabda hansı Məcəllənin hansı maddəsindən istifadə etdiyini qeyd et
3. Əgər cavab verilən maddələrdə yoxdursa, bunu açıq bildir
4. Sadə və aydın dildə izah et
5. Hüquqi məsləhət deyil, məlumat verdiyini qeyd et"""


class AzLexRAG:

    def __init__(self):
        # Load embedding model
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(MODEL_NAME)

        # Connect to ChromaDB
        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION)
        print(f"  {self.collection.count():,} chunks loaded")

        # Setup Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        self.gemini = genai.Client(api_key=api_key)
        print("  Gemini API ready\n")

    def retrieve(self, question, top_k=TOP_K):
        """Retrieve most relevant law chunks from ChromaDB"""
        embedding = self.embed_model.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            chunks.append({
                "text":     doc,
                "law_name": meta['law_name'],
                "chunk_id": meta['chunk_id'],
                "score":    round(1 - dist, 3)
            })

        return chunks

    def generate(self, question, chunks):
        context = ""
        for i, chunk in enumerate(chunks, 1):
            context += f"\n--- Mənbə {i}: {chunk['law_name']} ---\n"
            context += chunk['text'] + "\n"

        prompt = f"""{SYSTEM_PROMPT}

Aşağıdakı hüquqi maddələrə əsaslanaraq sualı cavablandır:

{context}

Sual: {question}

Cavab:"""

        response = self.gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text

    def ask(self, question):
        """Full RAG pipeline: retrieve + generate"""
        print(f"Question: {question}\n")

        chunks = self.retrieve(question)
        print(f"Retrieved {len(chunks)} relevant chunks:")
        for c in chunks:
            print(f"  [{c['score']}] {c['law_name']}")

        print("\nGenerating answer...\n")
        answer = self.generate(question, chunks)
        sources = [{"law": c['law_name'], "score": c['score']} for c in chunks]

        return {
            "question": question,
            "answer":   answer,
            "sources":  sources
        }


def interactive_mode(rag):
    print("AzLex — Azərbaycan Hüquq Assistenti")
    print("Çıxmaq üçün 'q' yazın")

    while True:
        question = input("Sualınız: ").strip()
        if question.lower() in ['q', 'quit', 'exit']:
            print("Görüşənədək!")
            break
        if not question:
            continue

        result = rag.ask(question)
        print("\n" + "=" * 60)
        print("CAVAB:")
        print(result['answer'])
        print("\nMƏNBƏLƏR:")
        for s in result['sources']:
            print(f"  • {s['law']} (uyğunluq: {s['score']})")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='AzLex RAG Pipeline')
    parser.add_argument('--question', '-q', type=str, help='Ask a single question')
    args = parser.parse_args()

    rag = AzLexRAG()

    if args.question:
        result = rag.ask(args.question)
        print("\n" + "=" * 60)
        print("CAVAB:")
        print(result['answer'])
        print("\nMƏNBƏLƏR:")
        for s in result['sources']:
            print(f"  • {s['law']} (uyğunluq: {s['score']})")
    else:
        interactive_mode(rag)


if __name__ == "__main__":
    main()