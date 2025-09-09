from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import torch

# 1. Define the documents and a query
documents = [
    "The sun is the star at the center of the Solar System.",
    "Mars, the Red Planet, is the fourth planet from the Sun.",
    "The Moon is Earth's only natural satellite.",
    "Jupiter is the largest planet in our Solar System.",
    "Saturn is known for its rings made of ice particles and rock.",
    "Venus is the second planet from the Sun and the hottest planet in our Solar System.",
    "Mercury is the smallest planet in the Solar System.",
]

query = "What is the biggest planet?"

# 2. Use Sentence-Transformers for semantic search
print("Loading semantic search model...")
# We use a small, efficient model suitable for this task.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for all documents
print("Creating document embeddings...")
document_embeddings = model.encode(documents, convert_to_tensor=True)

# Create an embedding for the query
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute cosine similarity between the query and all documents
cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]

# Find the top 3 results
top_semantic_results = torch.topk(cosine_scores, k=3)

# 3. Use BM25 for a baseline comparison
tokenized_corpus = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

print(f"\nSearching with BM25 for: '{query}'")
tokenized_query = query.split(" ")
# Get the scores for all documents at once
bm25_doc_scores = bm25.get_scores(tokenized_query)
# Find the top 3 results using torch
top_bm25_results = torch.topk(torch.tensor(bm25_doc_scores), k=3)

# 4. Print the results
print("\n--- Results from Sentence-Transformers (Semantic Search) ---")
for score, idx in zip(top_semantic_results.values, top_semantic_results.indices):
    print(f"Content: {documents[idx]}")
    print(f"Score: {score:.4f}")
    print("-" * 20)

print("\n--- Results from BM25 (Keyword Search) ---")
for score, idx in zip(top_bm25_results.values, top_bm25_results.indices):
    print(f"Content: {documents[idx]}")
    print(f"Score: {score:.4f}")
    print("-" * 20)
