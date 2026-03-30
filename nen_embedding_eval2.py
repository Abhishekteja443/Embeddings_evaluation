"""
NEN Schema Embedding Evaluation — v2
======================================
Models tested:
  PRE-TRAINED SENTENCE EMBEDDERS
  ├── BAAI/bge-large-en-v1.5
  ├── Qwen/Qwen3-Embedding-8B
  └── nvidia/llama-embed-nemotron-8b

  TRAINED-FROM-SCRATCH (on NEN sentences)
  ├── Word2Vec        (gensim)
  └── Doc2Vec / Sentence2Vec (gensim)

  KNOWLEDGE GRAPH EMBEDDINGS (on NEN triples)
  ├── TransE          (pykeen)
  └── TransR          (pykeen)

  ONTOLOGY EMBEDDING
  └── OWL2Vec*        (owl2vec-star)

Evaluation:
  - 10 Competency Questions derived from the NEN schema
  - Metrics: Hit@1, Hit@3, Hit@5, MRR, AvgCos_Relevant

Install before running:
  pip install torch transformers sentence-transformers
  pip install gensim
  pip install pykeen
  pip install owlready2 rdflib owl2vec-star
  pip install pandas numpy tabulate tqdm bitsandbytes accelerate

⚠️  Word2Vec / Doc2Vec / TransE / TransR are trained on 43 triples only.
    Expect lower scores — this is expected and informative, showing the
    value of pre-trained models on small domain datasets.

Usage:
  python3 nen_embedding_eval_v2.py
  python3 nen_embedding_eval_v2.py --owl path/to/ontology.owl
  python3 nen_embedding_eval_v2.py --skip-large   # skip 8B models
"""

import argparse
import os
import time
import warnings
import logging
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("nen_embedding_eval_v2.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE NEN DATA
# ══════════════════════════════════════════════════════════════════

NEN_CSV_PATH = "NEN.csv"

def load_nen(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    assert {"Subject", "Predicate", "Object"}.issubset(df.columns), \
        "CSV must have Subject, Predicate, Object columns"
    return df

PREDICATE_TEMPLATES = {
    "subClassOf":                    "{s} is a subclass of {o}.",
    "hasValue":                      "{s} has a value of type {o}.",
    "hasName":                       "{s} has a name which is a {o}.",
    "hasAgentRecord":                "{s} has an agent record called {o}.",
    "hasExternalReference":          "{s} has an external reference of type {o}.",
    "hasTemporalExtent":             "{s} has a temporal extent represented by {o}.",
    "temporalExtentContains":        "{s} contains temporal extent with value {o}.",
    "recordedAt":                    "{s} was recorded at an {o}.",
    "hasAgeValue":                   "{s} has an age value of type {o}.",
    "hasEventType":                  "{s} has an event type defined in {o}.",
    "providesParticipantRole":       "{s} provides a participant role of type {o}.",
    "subEventOf":                    "{s} is a sub-event of {o}.",
    "refersTo":                      "{s} refers to {o}.",
    "withID":                        "{s} is identified with an ID of type {o}.",
    "hasInterAgentRelationshipType": "{s} has an inter-agent relationship type from {o}.",
    "isRelationshipFrom":            "{s} is a relationship originating from {o}.",
    "isRelationshipTo":              "{s} is a relationship directed to {o}.",
    "hasNameVariant":                "{s} has a name variant of type {o}.",
    "hasPreferredNameVariant":       "{s} has a preferred name variant of type {o}.",
    "fullNameAsString":              "{s} stores a full name as a {o}.",
    "hasFirstnameAsString":          "{s} stores a first name as a {o}.",
    "hasSurnameAsString":            "{s} stores a surname as a {o}.",
    "hasParticipantRoleType":        "{s} has a participant role type from {o}.",
    "roleProvidedBy":                "{s} has its role provided by {o}.",
    "contains":                      "{s} contains {o}.",
    "endsAt":                        "{s} ends at {o}.",
    "startsAt":                      "{s} starts at {o}.",
    "fallsWithin":                   "{s} falls within {o}.",
    "occursAfter":                   "{s} occurs after {o}.",
    "occursBefore":                  "{s} occurs before {o}.",
}

def triples_to_sentences(df: pd.DataFrame) -> list[dict]:
    docs = []
    for _, row in df.iterrows():
        s, p, o = row["Subject"], row["Predicate"], row["Object"]
        tmpl = PREDICATE_TEMPLATES.get(p, "{s} {p} {o}.")
        sentence = tmpl.format(s=s, p=p, o=o)
        docs.append({"id": f"{s}|{p}|{o}", "sentence": sentence,
                     "subject": s, "predicate": p, "object": o})
    return docs

# ══════════════════════════════════════════════════════════════════
# 2. COMPETENCY QUESTIONS
# ══════════════════════════════════════════════════════════════════

COMPETENCY_QUESTIONS = [
    {
        "query": "What records does an Agent have?",
        "relevant": ["Agent|hasAgentRecord|AgentRecord",
                     "Agent|hasExternalReference|ExternalReference",
                     "Agent|hasName|xsd:string"],
    },
    {
        "query": "What is the structure of a NameRecord?",
        "relevant": ["NameRecord|hasNameVariant|NameVariant",
                     "NameRecord|hasPreferredNameVariant|NameVariant",
                     "NameRecord|subClassOf|AgentRecord"],
    },
    {
        "query": "How is time or date information modelled?",
        "relevant": ["TemporalExtent|startsAt|xsd:date",
                     "TemporalExtent|endsAt|xsd:date",
                     "TemporalExtent|contains|xsd:date",
                     "Timespan|startsAt|xsd:date",
                     "Timespan|endsAt|xsd:date",
                     "Timespan|subClassOf|TemporalExtent"],
    },
    {
        "query": "What properties does an Event have?",
        "relevant": ["Event|hasEventType|EventTypes.txt",
                     "Event|hasExternalReference|ExternalReference",
                     "Event|hasName|xsd:string",
                     "Event|hasTemporalExtent|TemporalExtent",
                     "Event|providesParticipantRole|ParticipantRoleRecord",
                     "Event|subEventOf|Event"],
    },
    {
        "query": "Which entity types are subclasses of AgentRecord?",
        "relevant": ["AgeRecord|subClassOf|AgentRecord",
                     "InterAgentRelationshipRecord|subClassOf|AgentRecord",
                     "OccupationRecord|subClassOf|AgentRecord",
                     "NameRecord|subClassOf|AgentRecord",
                     "ParticipantRoleRecord|subClassOf|AgentRecord",
                     "SexRecord|subClassOf|AgentRecord"],
    },
    {
        "query": "How are external identifiers or references stored?",
        "relevant": ["ExternalReference|refersTo|ExternalReferent",
                     "ExternalReference|withID|xsd:string",
                     "Agent|hasExternalReference|ExternalReference",
                     "Event|hasExternalReference|ExternalReference"],
    },
    {
        "query": "How is a person's occupation recorded?",
        "relevant": ["OccupationRecord|hasValue|Occupations.txt",
                     "OccupationRecord|subClassOf|AgentRecord"],
    },
    {
        "query": "What are the name components of a NameVariant?",
        "relevant": ["NameVariant|fullNameAsString|xsd:string",
                     "NameVariant|hasFirstnameAsString|xsd:string",
                     "NameVariant|hasSurnameAsString|xsd:string"],
    },
    {
        "query": "How are relationships between agents modelled?",
        "relevant": ["InterAgentRelationshipRecord|hasInterAgentRelationshipType|InterAgentRelationshipTypes.txt",
                     "InterAgentRelationshipRecord|isRelationshipFrom|Agent",
                     "InterAgentRelationshipRecord|isRelationshipTo|Agent",
                     "InterAgentRelationshipRecord|subClassOf|AgentRecord"],
    },
    {
        "query": "What role does a participant play in an event?",
        "relevant": ["ParticipantRoleRecord|hasParticipantRoleType|ParticipantRoleTypes.txt",
                     "ParticipantRoleRecord|roleProvidedBy|Event",
                     "ParticipantRoleRecord|subClassOf|AgentRecord",
                     "Event|providesParticipantRole|ParticipantRoleRecord"],
    },
]

# ══════════════════════════════════════════════════════════════════
# 3. SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T

def evaluate(sim_matrix: np.ndarray,
             doc_ids: list,
             questions: list,
             k_values: list = [1, 3, 5]) -> dict:
    id_to_idx   = {d: i for i, d in enumerate(doc_ids)}
    hits        = {k: [] for k in k_values}
    mrr_scores  = []
    avg_cos_rel = []

    for q_idx, cq in enumerate(questions):
        sims             = sim_matrix[q_idx]
        ranked           = np.argsort(sims)[::-1]
        relevant_ids     = set(cq["relevant"])
        relevant_indices = {id_to_idx[r] for r in relevant_ids if r in id_to_idx}

        for k in k_values:
            top_k = set(ranked[:k])
            hits[k].append(1.0 if top_k & relevant_indices else 0.0)

        rr = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if idx in relevant_indices:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

        rel_sims = [sims[i] for i in relevant_indices if i < len(sims)]
        avg_cos_rel.append(np.mean(rel_sims) if rel_sims else 0.0)

    return {
        **{f"Hit@{k}": np.mean(hits[k]) for k in k_values},
        "MRR":             np.mean(mrr_scores),
        "AvgCos_Relevant": np.mean(avg_cos_rel),
    }

def print_per_query(sim_matrix, doc_ids, doc_sentences):
    for q_idx, cq in enumerate(COMPETENCY_QUESTIONS):
        sims     = sim_matrix[q_idx]
        top3_idx = np.argsort(sims)[::-1][:3]
        logger.info("  CQ%02d: %s", q_idx + 1, cq["query"])
        for rank, idx in enumerate(top3_idx, 1):
            hit = "✓" if doc_ids[idx] in cq["relevant"] else " "
            logger.info("    %d. [%s] %-70s  sim=%.4f",
                        rank, hit, doc_sentences[idx][:70], sims[idx])
        logger.info("")

# ══════════════════════════════════════════════════════════════════
# 4A. PRE-TRAINED SENTENCE TRANSFORMERS  (bge, Qwen3, Nemotron)
# ══════════════════════════════════════════════════════════════════

def run_sentence_transformer(model_cfg, doc_sentences, doc_ids, results):
    from sentence_transformers import SentenceTransformer
    name     = model_cfg["name"]
    hf_id    = model_cfg["hf_id"]
    q_prefix = model_cfg.get("query_prefix", "")

    logger.info("─" * 65)
    logger.info("  MODEL: %s", name)
    t0 = time.time()
    try:
        model = SentenceTransformer(hf_id, trust_remote_code=True)
        logger.info("  Embedding schema triples ...")
        doc_emb = np.array(model.encode(doc_sentences, normalize_embeddings=True,
                                        show_progress_bar=True, batch_size=8))
        query_texts = [q_prefix + cq["query"] for cq in COMPETENCY_QUESTIONS]
        logger.info("  Embedding competency questions ...")
        q_emb = np.array(model.encode(query_texts, normalize_embeddings=True,
                                      show_progress_bar=True, batch_size=8))
        elapsed    = time.time() - t0
        sim_matrix = cosine_similarity_matrix(q_emb, doc_emb)
        metrics    = evaluate(sim_matrix, doc_ids, COMPETENCY_QUESTIONS)
        metrics.update({"Model": name, "Embed_dim": doc_emb.shape[1],
                        "Time_sec": round(elapsed, 1), "Type": "Pre-trained ST"})
        results.append(metrics)
        logger.info("  Results: %s", {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in metrics.items() if k != "Model"})
        print_per_query(sim_matrix, doc_ids, doc_sentences)
    except Exception as e:
        logger.error("  ✗ ERROR: %s", e)
        results.append({"Model": name, "Type": "Pre-trained ST", "Error": str(e)})

# ══════════════════════════════════════════════════════════════════
# 4B. WORD2VEC
# ══════════════════════════════════════════════════════════════════

def run_word2vec(doc_sentences, doc_ids, results):
    """
    Train Word2Vec on NEN sentences.
    Document vector = mean of word vectors in the sentence.
    Query vector    = mean of word vectors in the query.
    """
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess

    name = "Word2Vec"
    logger.info("─" * 65)
    logger.info("  MODEL: %s  (trained on NEN sentences)", name)
    logger.info("  ⚠  Only 43 training sentences — expect lower scores")
    t0 = time.time()
    try:
        tokenized = [simple_preprocess(s) for s in doc_sentences]

        # Train: more epochs to compensate for tiny corpus
        model = Word2Vec(
            sentences=tokenized,
            vector_size=300,
            window=5,
            min_count=1,       # keep all words (tiny corpus)
            workers=4,
            epochs=200,        # many passes over small data
            sg=1,              # skip-gram
        )

        def sentence_vector(tokens):
            vecs = [model.wv[t] for t in tokens if t in model.wv]
            return np.mean(vecs, axis=0) if vecs else np.zeros(300)

        doc_emb = np.array([sentence_vector(simple_preprocess(s))
                            for s in doc_sentences])
        q_emb   = np.array([sentence_vector(simple_preprocess(cq["query"]))
                            for cq in COMPETENCY_QUESTIONS])

        elapsed    = time.time() - t0
        sim_matrix = cosine_similarity_matrix(q_emb, doc_emb)
        metrics    = evaluate(sim_matrix, doc_ids, COMPETENCY_QUESTIONS)
        metrics.update({"Model": name, "Embed_dim": 300,
                        "Time_sec": round(elapsed, 1), "Type": "Trained (NEN only)"})
        results.append(metrics)
        logger.info("  Results: %s", {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in metrics.items() if k != "Model"})
        print_per_query(sim_matrix, doc_ids, doc_sentences)
    except Exception as e:
        logger.error("  ✗ ERROR: %s", e)
        results.append({"Model": name, "Type": "Trained (NEN only)", "Error": str(e)})

# ══════════════════════════════════════════════════════════════════
# 4C. DOC2VEC / SENTENCE2VEC
# ══════════════════════════════════════════════════════════════════

def run_doc2vec(doc_sentences, doc_ids, results):
    """
    Train Doc2Vec (PV-DM) on NEN sentences.
    Each sentence gets a dedicated document vector.
    Query vector is inferred at test time.
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from gensim.utils import simple_preprocess

    name = "Doc2Vec (Sentence2Vec)"
    logger.info("─" * 65)
    logger.info("  MODEL: %s  (trained on NEN sentences)", name)
    logger.info("  ⚠  Only 43 training sentences — expect lower scores")
    t0 = time.time()
    try:
        tagged = [TaggedDocument(words=simple_preprocess(s), tags=[i])
                  for i, s in enumerate(doc_sentences)]

        model = Doc2Vec(
            documents=tagged,
            vector_size=300,
            window=5,
            min_count=1,
            workers=4,
            epochs=200,
            dm=1,          # PV-DM mode (paragraph vector + word context)
        )

        doc_emb = np.array([model.dv[i] for i in range(len(doc_sentences))])

        # Infer query vectors (not seen during training)
        q_emb = np.array([
            model.infer_vector(simple_preprocess(cq["query"]), epochs=50)
            for cq in COMPETENCY_QUESTIONS
        ])

        elapsed    = time.time() - t0
        sim_matrix = cosine_similarity_matrix(q_emb, doc_emb)
        metrics    = evaluate(sim_matrix, doc_ids, COMPETENCY_QUESTIONS)
        metrics.update({"Model": name, "Embed_dim": 300,
                        "Time_sec": round(elapsed, 1), "Type": "Trained (NEN only)"})
        results.append(metrics)
        logger.info("  Results: %s", {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in metrics.items() if k != "Model"})
        print_per_query(sim_matrix, doc_ids, doc_sentences)
    except Exception as e:
        logger.error("  ✗ ERROR: %s", e)
        results.append({"Model": name, "Type": "Trained (NEN only)", "Error": str(e)})

# ══════════════════════════════════════════════════════════════════
# 4D. TRANSE & TRANSR  (pykeen)
# ══════════════════════════════════════════════════════════════════

def _triples_to_pykeen_format(df: pd.DataFrame):
    """Convert NEN dataframe to pykeen TriplesFactory."""
    from pykeen.triples import TriplesFactory
    triples = df[["Subject", "Predicate", "Object"]].values.astype(str)
    return TriplesFactory.from_labeled_triples(triples)

def _kg_embed_and_eval(model_name, tf, df, doc_ids, results):
    """
    Train a pykeen KGE model, then:
      - doc embedding  = entity vector for Subject of each triple
      - query matching = find triples whose Subject embedding is
                         closest to the mean embedding of query keywords
    
    Note on KGE evaluation design:
    TransE/TransR learn entity & relation vectors, not sentence vectors.
    We evaluate by:
      1. Representing each triple as: vec(Subject) + vec(Predicate) - vec(Object)
         (the TransE scoring intuition: h + r ≈ t)
      2. Representing queries as mean of entity vectors whose names
         appear as keywords in the query string.
      3. Computing cosine similarity between query vectors and triple vectors.
    """
    import torch
    from pykeen.pipeline import pipeline

    logger.info("─" * 65)
    logger.info("  MODEL: %s  (trained on NEN triples)", model_name)
    logger.info("  ⚠  Only 43 triples — expect lower scores vs pre-trained")
    t0 = time.time()
    try:
        result = pipeline(
            training=tf,
            testing=tf,          # same set (tiny corpus — no held-out split)
            model=model_name,
            training_kwargs=dict(num_epochs=500, batch_size=32),
            optimizer="Adam",
            optimizer_kwargs=dict(lr=0.01),
            random_seed=42,
            use_tqdm=False,
        )
        kge_model = result.model
        kge_model.eval()

        entity_repr  = kge_model.entity_representations[0]
        relation_repr = kge_model.relation_representations[0]

        with torch.no_grad():
            ent_emb = entity_repr(
                torch.arange(tf.num_entities)
            ).cpu().numpy()
            rel_emb = relation_repr(
                torch.arange(tf.num_relations)
            ).cpu().numpy()

        ent2idx = tf.entity_to_id
        rel2idx = tf.relation_to_id

        # Triple vectors: h + r - t  (TransE scoring)
        triple_vecs = []
        for _, row in df.iterrows():
            s, p, o = str(row["Subject"]), str(row["Predicate"]), str(row["Object"])
            h = ent_emb[ent2idx[s]] if s in ent2idx else np.zeros(ent_emb.shape[1])
            r = rel_emb[rel2idx[p]] if p in rel2idx else np.zeros(rel_emb.shape[1])
            t = ent_emb[ent2idx[o]] if o in ent2idx else np.zeros(ent_emb.shape[1])
            triple_vecs.append(h + r - t)
        doc_emb = np.array(triple_vecs)

        # Query vectors: mean of entity vectors for keywords found in query
        all_entities = list(ent2idx.keys())
        def query_to_vec(query_text):
            matched = [ent_emb[ent2idx[e]] for e in all_entities
                       if e.lower() in query_text.lower()]
            return np.mean(matched, axis=0) if matched else np.zeros(ent_emb.shape[1])

        q_emb = np.array([query_to_vec(cq["query"]) for cq in COMPETENCY_QUESTIONS])

        elapsed    = time.time() - t0
        sim_matrix = cosine_similarity_matrix(q_emb, doc_emb)
        metrics    = evaluate(sim_matrix, doc_ids, COMPETENCY_QUESTIONS)
        dim        = ent_emb.shape[1]
        metrics.update({"Model": model_name, "Embed_dim": dim,
                        "Time_sec": round(elapsed, 1), "Type": "KGE (NEN triples)"})
        results.append(metrics)
        logger.info("  Results: %s", {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in metrics.items() if k != "Model"})
        doc_sentences_kge = [f"{row.Subject} {row.Predicate} {row.Object}"
                             for _, row in df.iterrows()]
        print_per_query(sim_matrix, doc_ids, doc_sentences_kge)
    except Exception as e:
        logger.error("  ✗ ERROR: %s", e)
        results.append({"Model": model_name, "Type": "KGE (NEN triples)", "Error": str(e)})

def run_transe(df, doc_ids, results):
    tf = _triples_to_pykeen_format(df)
    _kg_embed_and_eval("TransE", tf, df, doc_ids, results)

def run_transr(df, doc_ids, results):
    tf = _triples_to_pykeen_format(df)
    _kg_embed_and_eval("TransR", tf, df, doc_ids, results)

# ══════════════════════════════════════════════════════════════════
# 4E. OWL2VEC*
# ══════════════════════════════════════════════════════════════════

def run_owl2vec(owl_path: str, df: pd.DataFrame, doc_ids: list, results: list):
    """
    OWL2Vec* pipeline:
      1. Parse OWL file with owlready2 / rdflib
      2. Generate random walks over the ontology graph
      3. Train Word2Vec on the walks (+ annotation text)
      4. Class/entity vector = Word2Vec vector for the IRI fragment
      5. Triple vector = mean(subject_vec, predicate_vec, object_vec)
      6. Query vector  = mean of entity vecs matching query keywords
    """
    name = "OWL2Vec*"
    logger.info("─" * 65)
    logger.info("  MODEL: %s  (OWL file: %s)", name, owl_path)
    t0 = time.time()
    try:
        # ── Try owl2vec-star library first ─────────────────────
        try:
            from owl2vec_star.owl2vec_star import OWL2VecStar
            logger.info("  Using owl2vec-star library ...")
            model = OWL2VecStar(
                ontology_file=owl_path,
                cache_dir="./owl2vec_cache",
                walk_depth=4,
                walk_number=100,
                embed_size=200,
                window=5,
                min_count=1,
                epochs=100,
            )
            model.train()
            entity_model = model.wv  # gensim KeyedVectors
        except ImportError:
            # ── Fallback: manual random walk + Word2Vec ─────────
            logger.info("  owl2vec-star not found — using manual RDF walk fallback ...")
            from rdflib import Graph, URIRef
            from gensim.models import Word2Vec
            import random

            g = Graph()
            g.parse(owl_path)
            logger.info("  Parsed OWL: %d triples", len(g))

            # Build adjacency for random walks
            adjacency = {}
            walks_corpus = []
            for s, p, o in g:
                s_str = str(s).split("#")[-1].split("/")[-1]
                p_str = str(p).split("#")[-1].split("/")[-1]
                o_str = str(o).split("#")[-1].split("/")[-1]
                if s_str not in adjacency:
                    adjacency[s_str] = []
                adjacency[s_str].append((p_str, o_str))
                walks_corpus.append([s_str, p_str, o_str])

            # Random walks
            all_nodes = list(adjacency.keys())
            for _ in range(200):
                node = random.choice(all_nodes)
                walk = [node]
                for _ in range(6):
                    if node in adjacency and adjacency[node]:
                        rel, nxt = random.choice(adjacency[node])
                        walk.extend([rel, nxt])
                        node = nxt
                    else:
                        break
                walks_corpus.append(walk)

            w2v = Word2Vec(
                sentences=walks_corpus,
                vector_size=200,
                window=5,
                min_count=1,
                workers=4,
                epochs=100,
                sg=1,
            )
            entity_model = w2v.wv

        # ── Build triple embeddings ─────────────────────────────
        def get_vec(token):
            # Try exact, then lowercase, then fragment
            for t in [token, token.lower(),
                      token.split("#")[-1], token.split("/")[-1]]:
                if t in entity_model:
                    return entity_model[t]
            return None

        triple_vecs = []
        for _, row in df.iterrows():
            parts = [str(row["Subject"]), str(row["Predicate"]), str(row["Object"])]
            vecs  = [v for v in (get_vec(p) for p in parts) if v is not None]
            triple_vecs.append(np.mean(vecs, axis=0) if vecs
                               else np.zeros(entity_model.vector_size))
        doc_emb = np.array(triple_vecs)

        # ── Query embeddings ────────────────────────────────────
        from gensim.utils import simple_preprocess
        def query_vec(query_text):
            tokens = simple_preprocess(query_text)
            vecs   = [v for v in (get_vec(t) for t in tokens) if v is not None]
            return np.mean(vecs, axis=0) if vecs else np.zeros(entity_model.vector_size)

        q_emb = np.array([query_vec(cq["query"]) for cq in COMPETENCY_QUESTIONS])

        elapsed    = time.time() - t0
        sim_matrix = cosine_similarity_matrix(q_emb, doc_emb)
        metrics    = evaluate(sim_matrix, doc_ids, COMPETENCY_QUESTIONS)
        metrics.update({"Model": name, "Embed_dim": entity_model.vector_size,
                        "Time_sec": round(elapsed, 1), "Type": "Ontology (OWL)"})
        results.append(metrics)
        logger.info("  Results: %s", {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in metrics.items() if k != "Model"})
        doc_sentences_owl = [f"{row.Subject} {row.Predicate} {row.Object}"
                             for _, row in df.iterrows()]
        print_per_query(sim_matrix, doc_ids, doc_sentences_owl)

    except Exception as e:
        logger.error("  ✗ ERROR running OWL2Vec*: %s", e)
        results.append({"Model": name, "Type": "Ontology (OWL)", "Error": str(e)})

# ══════════════════════════════════════════════════════════════════
# 5. MODEL REGISTRY  (pre-trained sentence transformers)
# ══════════════════════════════════════════════════════════════════

PRETRAINED_MODELS = [
    {
        "name":         "bge-large-en-v1.5",
        "hf_id":        "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    {
        "name":         "Qwen3-Embedding-8B",
        "hf_id":        "Qwen/Qwen3-Embedding-8B",
        "query_prefix": "Instruct: Given a schema query, retrieve the most relevant triples\nQuery: ",
    },
    {
        "name":         "llama-embed-nemotron-8b",
        "hf_id":        "nvidia/llama-embed-nemotron-8b",
        "query_prefix": "",
    },
]

# ══════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_evaluation(nen_path: str = NEN_CSV_PATH,
                   owl_path: str = None,
                   skip_large: bool = False):

    logger.info("=" * 65)
    logger.info("  NEN SCHEMA EMBEDDING EVALUATION  v2")
    logger.info("=" * 65)

    df            = load_nen(nen_path)
    docs          = triples_to_sentences(df)
    doc_sentences = [d["sentence"] for d in docs]
    doc_ids       = [d["id"]       for d in docs]
    results       = []

    logger.info("✓ Loaded %d triples | %d competency questions",
                len(docs), len(COMPETENCY_QUESTIONS))

    # ── Group 1: Pre-trained Sentence Transformers ───────────────
    logger.info("\n" + "═" * 65)
    logger.info("  GROUP 1 — Pre-trained Sentence Transformers")
    logger.info("═" * 65)
    for cfg in PRETRAINED_MODELS:
        if skip_large and cfg["name"] != "bge-large-en-v1.5":
            logger.info("  SKIPPING %s (--skip-large)", cfg["name"])
            continue
        run_sentence_transformer(cfg, doc_sentences, doc_ids, results)

    # ── Group 2: Trained-from-scratch text models ────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  GROUP 2 — Trained-from-scratch (NEN sentences only)")
    logger.info("═" * 65)
    run_word2vec(doc_sentences, doc_ids, results)
    run_doc2vec(doc_sentences, doc_ids, results)

    # ── Group 3: Knowledge Graph Embeddings ──────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  GROUP 3 — Knowledge Graph Embeddings (NEN triples)")
    logger.info("═" * 65)
    run_transe(df, doc_ids, results)
    run_transr(df, doc_ids, results)

    # ── Group 4: OWL2Vec* ────────────────────────────────────────
    if owl_path:
        logger.info("\n" + "═" * 65)
        logger.info("  GROUP 4 — Ontology Embedding (OWL2Vec*)")
        logger.info("═" * 65)
        run_owl2vec(owl_path, df, doc_ids, results)
    else:
        logger.info("\n  ℹ  No OWL file provided — skipping OWL2Vec*")
        logger.info("     Run with:  --owl path/to/ontology.owl")

    # ── Final Summary ─────────────────────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  FINAL COMPARISON SUMMARY")
    logger.info("═" * 65)

    headers = ["Model", "Type", "Hit@1", "Hit@3", "Hit@5",
               "MRR", "AvgCos_Rel", "Dim", "Time(s)"]
    rows = []
    for r in results:
        if "Error" in r:
            rows.append([r["Model"], r.get("Type", "?")]
                        + ["ERROR"] * (len(headers) - 2))
        else:
            rows.append([
                r["Model"],
                r.get("Type", "?"),
                f"{r['Hit@1']:.3f}",
                f"{r['Hit@3']:.3f}",
                f"{r['Hit@5']:.3f}",
                f"{r['MRR']:.3f}",
                f"{r['AvgCos_Relevant']:.4f}",
                r.get("Embed_dim", "?"),
                r.get("Time_sec", "?"),
            ])

    logger.info("\n%s", tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    logger.info("""
  Metric guide
  ────────────
  Hit@K          : Fraction of queries where ≥1 relevant triple is
                   in the top-K results  (higher = better, max 1.0)
  MRR            : Mean Reciprocal Rank — how high the FIRST correct
                   result ranks on average  (higher = better, max 1.0)
  AvgCos_Rel     : Mean cosine similarity of query vs its relevant triples
  Dim            : Embedding vector dimension
  Time(s)        : Wall-clock time for load + embed + train

  Model type key
  ──────────────
  Pre-trained ST      : Large models pre-trained on billions of sentences
  Trained (NEN only)  : Trained from scratch on 43 NEN sentences — SMALL DATA
  KGE (NEN triples)   : Knowledge graph embeddings on 43 triples   — SMALL DATA
  Ontology (OWL)      : OWL2Vec* over the full ontology graph
    """)

    out_path = "nen_eval_results_v2.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    logger.info("  ✓ Results saved to %s", out_path)


# ══════════════════════════════════════════════════════════════════
# 7. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NEN Embedding Evaluation — all 8 models")
    parser.add_argument("--nen",        default=NEN_CSV_PATH,
                        help="Path to NEN CSV (default: NEN.csv)")
    parser.add_argument("--owl",        default=None,
                        help="Path to OWL file for OWL2Vec* (optional)")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip Qwen3 and Nemotron 8B models")
    args = parser.parse_args()

    run_evaluation(
        nen_path   = args.nen,
        owl_path   = args.owl,
        skip_large = args.skip_large,
    )