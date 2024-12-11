import numpy as np

from jinaai_embeddings import JinaEmbeddings

docs = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog and the brown cat.",
    "The quick brown fox jumps over the lazy dog and the brown cat. The brown cat is sleeping.",
]

embeddings = JinaEmbeddings()

# Embed the documents
embs = embeddings.embed_documents(docs)
shape = np.array(embs).shape
print(shape)
