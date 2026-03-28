"""
NEN Schema Embedding Evaluation
================================
Models tested:
  - Qwen/Qwen3-Embedding-8B
  - nvidia/llama-embed-nemotron-8b
  - BAAI/bge-large-en-v1.5

Approach:
  1. Convert NEN triples → natural language sentences
  2. Embed with each model
  3. Evaluate using Competency Questions (CQs) derived from the schema
  4. Metrics: Cosine Similarity, Hit@K (K=1,3,5), MRR

Requirements (install before running):
  pip install torch transformers sentence-transformers pandas numpy scikit-learn tabulate tqdm
  # For Qwen3 & Nemotron (large models ~16GB each), ensure you have enough RAM/VRAM
  # or use quantized versions via bitsandbytes:
  pip install bitsandbytes accelerate
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD NEN SCHEMA
# ─────────────────────────────────────────────

NEN_CSV_PATH = "NEN.csv"   # ← change path if needed

def load_nen(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    assert {"Subject", "Predicate", "Object"}.issubset(df.columns), \
        "CSV must have Subject, Predicate, Object columns"
    return df

def triples_to_sentences(df: pd.DataFrame) -> list[dict]:
    """Convert each triple into a natural language sentence for embedding."""
    predicate_templates = {
        "subClassOf":                   "{s} is a subclass of {o}.",
        "hasValue":                     "{s} has a value of type {o}.",
        "hasName":                      "{s} has a name which is a {o}.",
        "hasAgentRecord":               "{s} has an agent record called {o}.",
        "hasExternalReference":         "{s} has an external reference of type {o}.",
        "hasTemporalExtent":            "{s} has a temporal extent represented by {o}.",
        "temporalExtentContains":       "{s} contains temporal extent with value {o}.",
        "recordedAt":                   "{s} was recorded at an {o}.",
        "hasAgeValue":                  "{s} has an age value of type {o}.",
        "hasEventType":                 "{s} has an event type defined in {o}.",
        "providesParticipantRole":      "{s} provides a participant role of type {o}.",
        "subEventOf":                   "{s} is a sub-event of {o}.",
        "refersTo":                     "{s} refers to {o}.",
        "withID":                       "{s} is identified with an ID of type {o}.",
        "hasInterAgentRelationshipType":"{s} has an inter-agent relationship type from {o}.",
        "isRelationshipFrom":           "{s} is a relationship originating from {o}.",
        "isRelationshipTo":             "{s} is a relationship directed to {o}.",
        "hasNameVariant":               "{s} has a name variant of type {o}.",
        "hasPreferredNameVariant":      "{s} has a preferred name variant of type {o}.",
        "fullNameAsString":             "{s} stores a full name as a {o}.",
        "hasFirstnameAsString":         "{s} stores a first name as a {o}.",
        "hasSurnameAsString":           "{s} stores a surname as a {o}.",
        "hasParticipantRoleType":       "{s} has a participant role type from {o}.",
        "roleProvidedBy":               "{s} has its role provided by {o}.",
        "contains":                     "{s} contains {o}.",
        "endsAt":                       "{s} ends at {o}.",
        "startsAt":                     "{s} starts at {o}.",
        "fallsWithin":                  "{s} falls within {o}.",
        "occursAfter":                  "{s} occurs after {o}.",
        "occursBefore":                 "{s} occurs before {o}.",
    }
    docs = []
    for _, row in df.iterrows():
        s, p, o = row["Subject"], row["Predicate"], row["Object"]
        tmpl = predicate_templates.get(p, "{s} {p} {o}.")
        sentence = tmpl.format(s=s, p=p, o=o)
        docs.append({"id": f"{s}|{p}|{o}", "sentence": sentence,
                     "subject": s, "predicate": p, "object": o})
    return docs

# ─────────────────────────────────────────────
# 2. COMPETENCY QUESTIONS  (query → expected subjects)
# ─────────────────────────────────────────────

COMPETENCY_QUESTIONS = [
    {
        "query":    "What records does an Agent have?",
        "relevant": ["Agent|hasAgentRecord|AgentRecord",
                     "Agent|hasExternalReference|ExternalReference",
                     "Agent|hasName|xsd:string"],
    },
    {
        "query":    "What is the structure of a NameRecord?",
        "relevant": ["NameRecord|hasNameVariant|NameVariant",
                     "NameRecord|hasPreferredNameVariant|NameVariant",
                     "NameRecord|subClassOf|AgentRecord"],
    },
    {
        "query":    "How is time or date information modelled?",
        "relevant": ["TemporalExtent|startsAt|xsd:date",
                     "TemporalExtent|endsAt|xsd:date",
                     "TemporalExtent|contains|xsd:date",
                     "Timespan|startsAt|xsd:date",
                     "Timespan|endsAt|xsd:date",
                     "Timespan|subClassOf|TemporalExtent"],
    },
    {
        "query":    "What properties does an Event have?",
        "relevant": ["Event|hasEventType|EventTypes.txt",
                     "Event|hasExternalReference|ExternalReference",
                     "Event|hasName|xsd:string",
                     "Event|hasTemporalExtent|TemporalExtent",
                     "Event|providesParticipantRole|ParticipantRoleRecord",
                     "Event|subEventOf|Event"],
    },
    {
        "query":    "Which entity types are subclasses of AgentRecord?",
        "relevant": ["AgeRecord|subClassOf|AgentRecord",
                     "InterAgentRelationshipRecord|subClassOf|AgentRecord",
                     "OccupationRecord|subClassOf|AgentRecord",
                     "NameRecord|subClassOf|AgentRecord",
                     "ParticipantRoleRecord|subClassOf|AgentRecord",
                     "SexRecord|subClassOf|AgentRecord"],
    },
    {
        "query":    "How are external identifiers or references stored?",
        "relevant": ["ExternalReference|refersTo|ExternalReferent",
                     "ExternalReference|withID|xsd:string",
                     "Agent|hasExternalReference|ExternalReference",
                     "Event|hasExternalReference|ExternalReference"],
    },
    {
        "query":    "How is a person's occupation recorded?",
        "relevant": ["OccupationRecord|hasValue|Occupations.txt",
                     "OccupationRecord|subClassOf|AgentRecord"],
    },
    {
        "query":    "What are the name components of a NameVariant?",
        "relevant": ["NameVariant|fullNameAsString|xsd:string",
                     "NameVariant|hasFirstnameAsString|xsd:string",
                     "NameVariant|hasSurnameAsString|xsd:string"],
    },
    {
        "query":    "How are relationships between agents modelled?",
        "relevant": ["InterAgentRelationshipRecord|hasInterAgentRelationshipType|InterAgentRelationshipTypes.txt",
                     "InterAgentRelationshipRecord|isRelationshipFrom|Agent",
                     "InterAgentRelationshipRecord|isRelationshipTo|Agent",
                     "InterAgentRelationshipRecord|subClassOf|AgentRecord"],
    },
    {
        "query":    "What role does a participant play in an event?",
        "relevant": ["ParticipantRoleRecord|hasParticipantRoleType|ParticipantRoleTypes.txt",
                     "ParticipantRoleRecord|roleProvidedBy|Event",
                     "ParticipantRoleRecord|subClassOf|AgentRecord",
                     "Event|providesParticipantRole|ParticipantRoleRecord"],
    },
]

# ─────────────────────────────────────────────
# 3. EMBEDDING HELPERS
# ─────────────────────────────────────────────

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of a and each row of b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T   # (n_queries, n_docs)


def load_sentence_transformer(model_name: str):
    """Load a model via sentence-transformers (works for bge and many HF models)."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {model_name} via SentenceTransformer ...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model

def embed_with_sentence_transformer(model, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=8)
    return np.array(embeddings)


def load_hf_model(model_name: str):
    """Load a causal/encoder model via HuggingFace transformers with optional 4-bit quant."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"  Loading {model_name} via HuggingFace transformers ...")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config,
                                          device_map="auto", trust_remote_code=True)
        print("  ✓ Loaded with 4-bit quantization")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, device_map="auto",
                                          torch_dtype="auto", trust_remote_code=True)
        print("  ✓ Loaded in full precision (no quantization)")
    model.eval()
    return tokenizer, model

def mean_pooling(model_output, attention_mask):
    import torch
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_with_hf(tokenizer, model, texts: list[str],
                  batch_size: int = 4) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding"):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        emb = mean_pooling(output, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().float().numpy())
    return np.vstack(all_embeddings)

# ─────────────────────────────────────────────
# 4. EVALUATION METRICS
# ─────────────────────────────────────────────

def evaluate(sim_matrix: np.ndarray,
             doc_ids: list[str],
             questions: list[dict],
             k_values: list[int] = [1, 3, 5]) -> dict:
    """
    sim_matrix : (n_queries, n_docs) cosine similarities
    doc_ids    : list of document IDs matching columns of sim_matrix
    questions  : list of CQ dicts with 'relevant' sets
    """
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    hits = {k: [] for k in k_values}
    mrr_scores = []
    avg_cos_relevant = []

    for q_idx, cq in enumerate(questions):
        sims = sim_matrix[q_idx]                          # (n_docs,)
        ranked = np.argsort(sims)[::-1]                   # descending
        relevant_ids = set(cq["relevant"])
        relevant_indices = {id_to_idx[r] for r in relevant_ids if r in id_to_idx}

        # Hit@K
        for k in k_values:
            top_k = set(ranked[:k])
            hits[k].append(1.0 if top_k & relevant_indices else 0.0)

        # MRR
        rr = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if idx in relevant_indices:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

        # Average cosine similarity of relevant docs
        rel_sims = [sims[i] for i in relevant_indices if i < len(sims)]
        avg_cos_relevant.append(np.mean(rel_sims) if rel_sims else 0.0)

    return {
        **{f"Hit@{k}": np.mean(hits[k]) for k in k_values},
        "MRR":              np.mean(mrr_scores),
        "AvgCos_Relevant":  np.mean(avg_cos_relevant),
    }

# ─────────────────────────────────────────────
# 5. MODEL REGISTRY
# ─────────────────────────────────────────────

MODELS = [
    {
        "name":    "bge-large-en-v1.5",
        "hf_id":   "BAAI/bge-large-en-v1.5",
        "backend": "sentence_transformer",
        # bge benefits from a query prefix
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    {
        "name":    "Qwen3-Embedding-8B",
        "hf_id":   "Qwen/Qwen3-Embedding-8B",
        "backend": "sentence_transformer",   # officially supports ST
        "query_prefix": "Instruct: Given a schema query, retrieve the most relevant triples\nQuery: ",
    },
    {
        "name":    "llama-embed-nemotron-8b",
        "hf_id":   "nvidia/llama-embed-nemotron-8b",
        "backend": "sentence_transformer",
        "query_prefix": "",
    },
]

# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_evaluation(nen_path: str = NEN_CSV_PATH):
    print("=" * 65)
    print("  NEN SCHEMA EMBEDDING EVALUATION")
    print("=" * 65)

    # ── Load & prepare data ──────────────────────────────────────
    df = load_nen(nen_path)
    docs = triples_to_sentences(df)
    doc_sentences = [d["sentence"] for d in docs]
    doc_ids       = [d["id"]       for d in docs]

    print(f"\n✓ Loaded {len(docs)} triples from NEN schema")
    print(f"✓ Prepared {len(COMPETENCY_QUESTIONS)} competency questions\n")

    # Preview queries
    print("── Competency Questions (Test Queries) ──────────────────")
    for i, cq in enumerate(COMPETENCY_QUESTIONS, 1):
        print(f"  CQ{i:02d}: {cq['query']}")
    print()

    results_summary = []

    for model_cfg in MODELS:
        name    = model_cfg["name"]
        hf_id   = model_cfg["hf_id"]
        backend = model_cfg["backend"]
        q_prefix = model_cfg.get("query_prefix", "")

        print("─" * 65)
        print(f"  MODEL: {name}")
        print("─" * 65)

        t0 = time.time()

        try:
            if backend == "sentence_transformer":
                model = load_sentence_transformer(hf_id)

                # Embed documents
                print("  Embedding schema triples ...")
                doc_embeddings = embed_with_sentence_transformer(model, doc_sentences)

                # Embed queries (with optional prefix)
                query_texts = [q_prefix + cq["query"] for cq in COMPETENCY_QUESTIONS]
                print("  Embedding competency questions ...")
                query_embeddings = embed_with_sentence_transformer(model, query_texts)

            else:   # hf fallback
                tokenizer, model = load_hf_model(hf_id)
                print("  Embedding schema triples ...")
                doc_embeddings = embed_with_hf(tokenizer, model, doc_sentences)
                query_texts = [q_prefix + cq["query"] for cq in COMPETENCY_QUESTIONS]
                print("  Embedding competency questions ...")
                query_embeddings = embed_with_hf(tokenizer, model, query_texts)

            elapsed = time.time() - t0

            # Cosine similarity matrix  (n_queries × n_docs)
            sim_matrix = cosine_similarity_matrix(query_embeddings, doc_embeddings)

            # Metrics
            metrics = evaluate(sim_matrix, doc_ids, COMPETENCY_QUESTIONS)
            metrics["Model"]         = name
            metrics["Embed_dim"]     = doc_embeddings.shape[1]
            metrics["Time_sec"]      = round(elapsed, 1)
            results_summary.append(metrics)

            print(f"\n  Results for {name}:")
            for k, v in metrics.items():
                if k not in ("Model",):
                    print(f"    {k:<22} {v if isinstance(v, (int,str)) else f'{v:.4f}'}")

            # ── Per-query detail ──────────────────────────────────
            print(f"\n  Per-Query Cosine Similarity (top-3 retrieved):")
            for q_idx, cq in enumerate(COMPETENCY_QUESTIONS):
                sims = sim_matrix[q_idx]
                top3_idx = np.argsort(sims)[::-1][:3]
                print(f"\n  CQ{q_idx+1:02d}: {cq['query']}")
                for rank, idx in enumerate(top3_idx, 1):
                    hit = "✓" if doc_ids[idx] in cq["relevant"] else " "
                    print(f"    {rank}. [{hit}] {doc_sentences[idx][:70]:<70}  sim={sims[idx]:.4f}")

        except Exception as e:
            print(f"  ✗ ERROR loading/running {name}: {e}")
            results_summary.append({"Model": name, "Error": str(e)})

        print()

    # ─────────────────────────────────────────────────────────────
    # 7. FINAL SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 65)

    table_rows = []
    headers = ["Model", "Hit@1", "Hit@3", "Hit@5", "MRR", "AvgCos_Relevant", "Embed_dim", "Time(s)"]

    for r in results_summary:
        if "Error" in r:
            table_rows.append([r["Model"]] + ["ERROR"] * (len(headers) - 1))
        else:
            table_rows.append([
                r["Model"],
                f"{r['Hit@1']:.3f}",
                f"{r['Hit@3']:.3f}",
                f"{r['Hit@5']:.3f}",
                f"{r['MRR']:.3f}",
                f"{r['AvgCos_Relevant']:.4f}",
                r["Embed_dim"],
                r["Time_sec"],
            ])

    print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))

    print("""
  Metric guide
  ────────────
  Hit@K          : Fraction of queries where ≥1 relevant triple
                   appears in the top-K retrieved results (higher = better)
  MRR            : Mean Reciprocal Rank — how high the first correct
                   result ranks on average (higher = better, max 1.0)
  AvgCos_Relevant: Average cosine similarity between a query and its
                   known-relevant triples (higher = better)
  Embed_dim      : Output vector dimension of the model
  Time(s)        : Total wall-clock time for loading + embedding
    """)

    # Save results CSV
    out_path = "nen_eval_results.csv"
    pd.DataFrame(results_summary).to_csv(out_path, index=False)
    print(f"  ✓ Results saved to {out_path}")


if __name__ == "__main__":
    run_evaluation()
