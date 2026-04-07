import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# -----------------------------
# Load Data
# -----------------------------
with open("CQs.txt") as f:
    CQs = [line.strip() for line in f if line.strip()]

with open("enslaved-v2.txt") as f:
    classes = [line.strip() for line in f if line.strip()]

CQs = [cq.lower() for cq in CQs]
classes = [c.lower() for c in classes]

# -----------------------------
# Convert classes to sentences
# -----------------------------
def convert_class_to_sentence(cls):
    return f"{cls} represents a concept or entity in a knowledge graph"

class_sentences = [convert_class_to_sentence(c) for c in classes]

# -----------------------------
# HF embedding helper
# -----------------------------
def hf_embed(texts, model, tokenizer, device="cuda"):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

# -----------------------------
# Models
# -----------------------------
models = {
    "bge-large": SentenceTransformer("BAAI/bge-large-en-v1.5"),
}

hf_models = {
    "bert": "bert-base-uncased",
    "qwen": "Qwen/Qwen3-Embedding-8B",
    "nvidia": "nvidia/llama-embed-nemotron-8b"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

hf_loaded = {}
for name, path in hf_models.items():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device)
    model.eval()
    hf_loaded[name] = (tokenizer, model)

# -----------------------------
# Precompute class embeddings
# -----------------------------
class_embeddings = {}

for name, model in models.items():
    emb = model.encode(class_sentences, normalize_embeddings=True)
    class_embeddings[name] = emb

for name, (tokenizer, model) in hf_loaded.items():
    emb = hf_embed(class_sentences, model, tokenizer, device)
    class_embeddings[name] = emb

# -----------------------------
# Run retrieval
# -----------------------------
TOP_K = 5
results = []

for cq in tqdm(CQs):
    row = {"CQ": cq}
    
    for model_name in class_embeddings:
        if model_name in models:
            cq_emb = models[model_name].encode([cq], normalize_embeddings=True)
        else:
            tokenizer, model = hf_loaded[model_name]
            cq_emb = hf_embed([cq], model, tokenizer, device)
        
        sims = cosine_similarity(cq_emb, class_embeddings[model_name])[0]
        top_idx = sims.argsort()[::-1][:TOP_K]
        
        top_classes = [(classes[i], float(sims[i])) for i in top_idx]
        row[model_name] = top_classes
    
    results.append(row)

# -----------------------------
# Save results
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("results_sentence_embedding.csv", index=False)

# Pretty print
for r in results[:5]:
    print("\nCQ:", r["CQ"])
    for m in class_embeddings:
        print(f"\nModel: {m}")
        for cls, score in r[m]:
            print(f"{cls} -> {score:.4f}")