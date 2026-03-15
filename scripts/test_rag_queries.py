"""Test RAG retrieval with sample queries."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'oasis-rag'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from indexer import load_index
from retriever import Retriever

QUERIES = [
    "snake bit him on the ankle",
    "hypothermia stopped shivering",
    "lightning storm no shelter",
    "caught in avalanche buried",
    "severe bleeding from arm",
]

def short_snippet(text: str, max_len: int = 300) -> str:
    text = text.replace('\n', ' ').strip()
    return text[:max_len] + ('…' if len(text) > max_len else '')

def main():
    print("Loading index …", flush=True)
    store = load_index()
    retriever = Retriever(store)
    print(f"Index loaded: {store.chunk_count} chunks\n")
    print("=" * 70)

    for query in QUERIES:
        print(f'\nQUERY: "{query}"')
        print("-" * 70)
        result = retriever.retrieve(query)
        chunks = result.chunks
        if not chunks:
            print("  (no results)")
        for rank, c in enumerate(chunks[:3], 1):
            print(f"  [{rank}] hybrid={c.hybrid_score:.4f}  cosine={c.cosine_score:.4f}  source={c.source}")
            print(f"       section: {c.section or '—'}")
            print(f"       {short_snippet(c.compressed_text or c.text)}")
        print(f"  [latency: {result.latency_ms:.1f}ms | stage1={result.stage1_count} | stage2={result.stage2_count}]")
        print()
    print("=" * 70)

if __name__ == '__main__':
    main()
