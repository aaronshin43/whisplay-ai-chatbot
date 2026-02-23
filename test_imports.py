print("Testing imports...")
try:
    import numpy
    print("numpy imported")
    import faiss
    print("faiss imported")
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers imported")
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded")
except Exception as e:
    print(f"Error: {e}")
