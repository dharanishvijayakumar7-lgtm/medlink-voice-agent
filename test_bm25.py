import json
from rank_bm25 import BM25Okapi

# Load medicines
with open('indian_otc_medicines_final.json', 'r', encoding='utf-8') as f:
    medicines = json.load(f)

print(f"Loaded {len(medicines)} medicines\n")

# Create documents
documents = []
medicine_map = {}


for idx, medicine in enumerate(medicines):
    name = medicine.get('name', '')
    composition = ' '.join(medicine.get('composition', []))
    indications = ' '.join(medicine.get('indications', []))
    therapeutic = medicine.get('therapeutic_class', '')
    
    doc = f"{name} {composition} {indications} {indications} {indications} {therapeutic}".lower()
    documents.append(doc)
    medicine_map[idx] = medicine

# Initialize BM25
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

print("BM25 initialized\n")

# Test queries
test_queries = ["fever", "headache", "cough", "acidity"]

for query in test_queries:
    print(f"Query: '{query}'")
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    
    for rank, idx in enumerate(top_indices, 1):
        med = medicine_map[idx]
        print(f"  {rank}. {med['name']} - {med['indications'][0] if med['indications'] else 'N/A'}")
    print()