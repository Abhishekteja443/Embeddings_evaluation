"""
Enslaved Ontology — Class Retrieval Evaluation
================================================
Given each Competency Question (CQ), rank all 44 ontology classes
by semantic similarity and store scores.

Models:
  PRE-TRAINED DENSE
  ├── bge-large-en-v1.5          (sentence-transformers)
  ├── Qwen3-Embedding-8B         (sentence-transformers)
  └── llama-embed-nemotron-8b    (sentence-transformers)

  SPARSE / KEYWORD
  └── TF-IDF                     (sklearn — fast baseline)

  SPARSE NEURAL
  └── SPLADE                     (naver/splade-cocondenser-selfdistil)

  LATE INTERACTION
  └── ColBERT                    (colbert-ir/colbertv2.0)

  DOMAIN-SPECIFIC
  └── BioWordVec                 (biomedical Word2Vec, 200d)

CQ pre-processing:
  - Stopwords removed
  - Placeholder tokens (X, XXX, XXXX) removed
  - CamelCase class names split into words for embedding

Output:
  - enslaved_class_scores.xlsx   one sheet per model,
                                 rows=CQs, cols=classes, values=similarity
  - enslaved_class_scores.csv    long format (CQ, Model, Class, Score)
  - enslaved_class_eval.log      full run log

Install:
  pip install sentence-transformers transformers torch
  pip install scikit-learn pandas openpyxl nltk tqdm
  pip install gensim            # for BioWordVec
  # ColBERT:
  pip install colbert-ai        # or: pip install ragatouille
  # SPLADE: no extra install needed (uses transformers)

BioWordVec model download (run once):
  Download bio_embedding_intrinsic from:
  https://ftp.ncbi.nlm.nih.gov/pub/lu/Supmat/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
  Place it at: ./BioWordVec_PubMed_MIMICIII_d200.bin

Usage:
  python3 enslaved_class_eval.py
  python3 enslaved_class_eval.py --skip-large        # skip 8B models
  python3 enslaved_class_eval.py --biowordvec path/to/BioWordVec.bin
"""

import argparse
import re
import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enslaved_class_eval.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# PATHS  (change these if your files are elsewhere)
# ══════════════════════════════════════════════════════════════
CLASSES_FILE = "enslaved-v2.txt"
CQS_FILE     = "CQs.txt"
OUTPUT_XLSX  = "enslaved_class_scores.xlsx"
OUTPUT_CSV   = "enslaved_class_scores.csv"
BIOWORDVEC_PATH = "./BioWordVec_PubMed_MIMICIII_d200.bin"

# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING & PRE-PROCESSING
# ══════════════════════════════════════════════════════════════

def load_classes(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [l.strip() for l in lines if l.strip()]

def load_cqs(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [l.strip() for l in lines if l.strip()]

def split_camel(name: str) -> str:
    """AgeRecord → Age Record | ECVO → ECVO | PlaceTypeCV → Place Type CV"""
    # Insert space before each uppercase letter that follows a lowercase letter
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # Insert space before uppercase sequences followed by lowercase
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.strip()

def clean_cq(cq: str) -> str:
    """
    Remove stopwords and placeholder tokens from a CQ so only
    semantically meaningful words remain for matching.
    """
    import nltk
    try:
        stopwords = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stopwords = set(nltk.corpus.stopwords.words("english"))

    # Remove placeholder tokens like X, XX, XXX, XXXX
    cq_clean = re.sub(r"\b[Xx]+\b", "", cq)
    # Tokenise and strip stopwords + punctuation
    tokens = re.findall(r"[a-zA-Z]+", cq_clean)
    tokens = [t for t in tokens
              if t.lower() not in stopwords and len(t) > 1]
    return " ".join(tokens)

# ══════════════════════════════════════════════════════════════
# 2. SIMILARITY UTILITIES
# ══════════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,d)  b: (m,d)  → (n,m) cosine similarities"""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a @ b.T

def dot_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot product similarity — used for SPLADE sparse vectors"""
    return a @ b.T

# ══════════════════════════════════════════════════════════════
# 3. MODEL RUNNERS
# Each returns a (n_cqs × n_classes) similarity matrix
# ══════════════════════════════════════════════════════════════

# ── 3a. Dense sentence transformers ──────────────────────────

def run_dense_st(model_hf_id: str, model_name: str,
                 class_texts: list[str], cq_texts: list[str],
                 query_prefix: str = "") -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("  Loading %s ...", model_name)
        model = SentenceTransformer(model_hf_id, trust_remote_code=True)
        logger.info("  Embedding %d classes ...", len(class_texts))
        class_emb = np.array(model.encode(class_texts, normalize_embeddings=True,
                                          show_progress_bar=True, batch_size=16))
        queries = [query_prefix + q for q in cq_texts]
        logger.info("  Embedding %d CQs ...", len(queries))
        cq_emb = np.array(model.encode(queries, normalize_embeddings=True,
                                       show_progress_bar=True, batch_size=16))
        return cosine_sim(cq_emb, class_emb)
    except Exception as e:
        logger.error("  ✗ %s failed: %s", model_name, e)
        return None

# ── 3b. TF-IDF baseline ───────────────────────────────────────

def run_tfidf(class_texts: list[str], cq_texts: list[str]) -> np.ndarray | None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        logger.info("  Fitting TF-IDF on classes + CQs ...")
        all_texts = class_texts + cq_texts
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                              sublinear_tf=True)
        vec.fit(all_texts)
        class_mat = vec.transform(class_texts).toarray()
        cq_mat    = vec.transform(cq_texts).toarray()
        return cosine_similarity(cq_mat, class_mat)
    except Exception as e:
        logger.error("  ✗ TF-IDF failed: %s", e)
        return None

# ── 3c. SPLADE (sparse neural retrieval) ─────────────────────

def run_splade(class_texts: list[str], cq_texts: list[str]) -> np.ndarray | None:
    """
    Uses naver/splade-cocondenser-selfdistil.
    Encodes text as sparse vectors over the full vocabulary.
    Similarity = dot product of sparse vectors.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        logger.info("  Loading SPLADE ...")
        model_id  = "naver/splade-cocondenser-selfdistil"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = AutoModelForMaskedLM.from_pretrained(model_id)
        model.eval()

        def splade_encode(texts: list[str], batch_size=8) -> np.ndarray:
            all_vecs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                enc   = tokenizer(batch, return_tensors="pt",
                                  padding=True, truncation=True, max_length=256)
                with torch.no_grad():
                    logits = model(**enc).logits           # (B, seq, vocab)
                # SPLADE aggregation: max over tokens, ReLU + log(1+x)
                sparse = torch.log(1 + torch.relu(logits))
                sparse = sparse.max(dim=1).values          # (B, vocab)
                all_vecs.append(sparse.cpu().numpy())
            return np.vstack(all_vecs)

        logger.info("  SPLADE encoding %d classes ...", len(class_texts))
        class_vecs = splade_encode(class_texts)
        logger.info("  SPLADE encoding %d CQs ...", len(cq_texts))
        cq_vecs    = splade_encode(cq_texts)
        return dot_sim(cq_vecs, class_vecs)
    except Exception as e:
        logger.error("  ✗ SPLADE failed: %s", e)
        return None

# ── 3d. ColBERT (late interaction) ───────────────────────────

def run_colbert(class_texts: list[str], cq_texts: list[str]) -> np.ndarray | None:
    """
    ColBERT late interaction:
      score(q,d) = Σ_qi  max_dj  (qi · dj)
    Uses colbert-ir/colbertv2.0 via transformers directly
    (no ragatouille dependency needed).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        logger.info("  Loading ColBERT ...")
        model_id  = "colbert-ir/colbertv2.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = AutoModel.from_pretrained(model_id)
        model.eval()

        def colbert_encode(texts: list[str], batch_size=8) -> list[np.ndarray]:
            """Returns list of (seq_len, dim) token embeddings per text."""
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                enc   = tokenizer(batch, return_tensors="pt",
                                  padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    out = model(**enc).last_hidden_state  # (B, seq, dim)
                # L2 normalise each token vector
                out = out / (out.norm(dim=-1, keepdim=True) + 1e-10)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                for b in range(out.shape[0]):
                    # only keep non-padding token embeddings
                    length = int(enc["attention_mask"][b].sum())
                    all_embs.append(out[b, :length, :].cpu().numpy())
            return all_embs

        logger.info("  ColBERT encoding %d classes ...", len(class_texts))
        class_embs = colbert_encode(class_texts)
        logger.info("  ColBERT encoding %d CQs ...", len(cq_texts))
        cq_embs    = colbert_encode(cq_texts)

        # Late interaction: MaxSim
        sim_matrix = np.zeros((len(cq_embs), len(class_embs)))
        for qi, q_toks in enumerate(cq_embs):          # q_toks: (Lq, dim)
            for di, d_toks in enumerate(class_embs):   # d_toks: (Ld, dim)
                # (Lq, Ld) dot products → max over doc tokens → sum over query tokens
                scores = q_toks @ d_toks.T             # (Lq, Ld)
                sim_matrix[qi, di] = float(scores.max(axis=1).sum())

        # Normalise to [0,1] range for comparability
        mn, mx = sim_matrix.min(), sim_matrix.max()
        if mx > mn:
            sim_matrix = (sim_matrix - mn) / (mx - mn)
        return sim_matrix
    except Exception as e:
        logger.error("  ✗ ColBERT failed: %s", e)
        return None

# ── 3e. BioWordVec ────────────────────────────────────────────

def run_biowordvec(bio_path: str,
                   class_texts: list[str],
                   cq_texts: list[str]) -> np.ndarray | None:
    """
    BioWordVec: biomedical Word2Vec trained on PubMed + MIMIC-III.
    Document vector = mean of word vectors.
    Download: https://ftp.ncbi.nlm.nih.gov/pub/lu/Supmat/BioSentVec/
              BioWordVec_PubMed_MIMICIII_d200.bin
    """
    if not os.path.exists(bio_path):
        logger.warning("  BioWordVec binary not found at %s — skipping", bio_path)
        logger.warning("  Download from: https://ftp.ncbi.nlm.nih.gov/pub/lu/"
                       "Supmat/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin")
        return None
    try:
        import gensim
        logger.info("  Loading BioWordVec from %s ...", bio_path)
        wv = gensim.models.KeyedVectors.load_word2vec_format(bio_path, binary=True)
        dim = wv.vector_size

        def text_vec(text: str) -> np.ndarray:
            tokens = re.findall(r"[a-zA-Z]+", text.lower())
            vecs   = [wv[t] for t in tokens if t in wv]
            return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

        logger.info("  Encoding %d classes ...", len(class_texts))
        class_emb = np.array([text_vec(t) for t in class_texts])
        logger.info("  Encoding %d CQs ...", len(cq_texts))
        cq_emb    = np.array([text_vec(t) for t in cq_texts])
        return cosine_sim(cq_emb, class_emb)
    except Exception as e:
        logger.error("  ✗ BioWordVec failed: %s", e)
        return None

# ══════════════════════════════════════════════════════════════
# 4. OUTPUT BUILDERS
# ══════════════════════════════════════════════════════════════

def build_outputs(all_results: dict,
                  classes: list[str],
                  cqs: list[str],
                  cqs_clean: list[str]):
    """
    all_results: {model_name: (n_cqs, n_classes) sim matrix or None}
    Saves:
      - enslaved_class_scores.xlsx  (one sheet per model)
      - enslaved_class_scores.csv   (long format)
    """
    long_rows = []

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for model_name, sim_matrix in all_results.items():
            if sim_matrix is None:
                logger.warning("  Skipping %s — no results", model_name)
                continue

            # ── Wide format sheet ─────────────────────────────
            df_wide = pd.DataFrame(
                np.round(sim_matrix, 6),
                index=cqs,
                columns=classes,
            )
            df_wide.index.name = "CQ"

            # Add a "top_5_classes" helper column
            top5 = []
            for row_idx in range(len(cqs)):
                scores = sim_matrix[row_idx]
                ranked = np.argsort(scores)[::-1][:5]
                top5.append(", ".join(
                    f"{classes[i]} ({scores[i]:.3f})" for i in ranked))
            df_wide.insert(0, "Top5_Classes", top5)

            sheet_name = model_name[:31]  # Excel sheet name limit
            df_wide.to_excel(writer, sheet_name=sheet_name)
            logger.info("  Wrote sheet: %s", sheet_name)

            # ── Long format for CSV ───────────────────────────
            for cq_idx, cq in enumerate(cqs):
                for cls_idx, cls in enumerate(classes):
                    long_rows.append({
                        "Model":        model_name,
                        "CQ_index":     cq_idx + 1,
                        "CQ_original":  cq,
                        "CQ_cleaned":   cqs_clean[cq_idx],
                        "Class":        cls,
                        "Class_split":  split_camel(cls),
                        "Similarity":   round(float(sim_matrix[cq_idx, cls_idx]), 6),
                    })

    df_long = pd.DataFrame(long_rows)
    # Sort: Model → CQ → descending similarity
    df_long = df_long.sort_values(
        ["Model", "CQ_index", "Similarity"], ascending=[True, True, False]
    ).reset_index(drop=True)
    df_long.to_csv(OUTPUT_CSV, index=False)
    logger.info("  ✓ Saved %s  (%d rows)", OUTPUT_CSV, len(df_long))
    logger.info("  ✓ Saved %s", OUTPUT_XLSX)
    return df_long

def print_sample(df_long: pd.DataFrame,
                 classes: list[str],
                 cqs: list[str],
                 n_cqs: int = 3):
    """Print top-5 classes for the first n_cqs CQs per model."""
    models = df_long["Model"].unique()
    for model in models:
        logger.info("\n  ── %s ──", model)
        df_m = df_long[df_long["Model"] == model]
        for cq_idx in range(1, n_cqs + 1):
            df_cq = df_m[df_m["CQ_index"] == cq_idx].head(5)
            cq_txt = df_cq["CQ_original"].iloc[0] if len(df_cq) else ""
            logger.info("  CQ%02d: %s", cq_idx, cq_txt)
            for _, row in df_cq.iterrows():
                logger.info("    %-30s  %.4f", row["Class"], row["Similarity"])

# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════

def main(classes_file=CLASSES_FILE,
         cqs_file=CQS_FILE,
         skip_large=False,
         bio_path=BIOWORDVEC_PATH):

    logger.info("=" * 65)
    logger.info("  ENSLAVED ONTOLOGY — CLASS RETRIEVAL EVALUATION")
    logger.info("=" * 65)

    # ── Load data ─────────────────────────────────────────────
    classes   = load_classes(classes_file)
    cqs_raw   = load_cqs(cqs_file)

    # Split camelCase for embedding
    class_texts = [split_camel(c) for c in classes]

    # Clean CQs: remove stopwords + placeholders
    import nltk
    nltk.download("stopwords", quiet=True)
    cqs_clean   = [clean_cq(cq) for cq in cqs_raw]
    # For dense models use full CQ (they handle stopwords fine);
    # for TF-IDF/BioWordVec use cleaned version
    cqs_full    = cqs_raw

    logger.info("✓ Classes: %d", len(classes))
    logger.info("✓ CQs: %d", len(cqs_raw))
    logger.info("\nSample class splitting:")
    for c, ct in zip(classes[:5], class_texts[:5]):
        logger.info("  %-35s → %s", c, ct)
    logger.info("\nSample CQ cleaning:")
    for raw, clean in zip(cqs_raw[:3], cqs_clean[:3]):
        logger.info("  RAW  : %s", raw)
        logger.info("  CLEAN: %s", clean)
        logger.info("")

    all_results = {}

    # ── Group 1: Dense Pre-trained ────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  GROUP 1 — Dense Pre-trained Sentence Transformers")
    logger.info("═" * 65)

    dense_models = [
        {
            "name":         "bge-large-en-v1.5",
            "hf_id":        "BAAI/bge-large-en-v1.5",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
        },
        {
            "name":         "Qwen3-Embedding-8B",
            "hf_id":        "Qwen/Qwen3-Embedding-8B",
            "query_prefix": ("Instruct: Given an ontology query about enslaved people, "
                             "retrieve the most relevant ontology classes\nQuery: "),
        },
        {
            "name":         "llama-embed-nemotron-8b",
            "hf_id":        "nvidia/llama-embed-nemotron-8b",
            "query_prefix": "",
        },
    ]

    for cfg in dense_models:
        if skip_large and "8B" in cfg["name"] or "nemotron" in cfg["name"]:
            logger.info("  SKIPPING %s (--skip-large)", cfg["name"])
            continue
        t0  = time.time()
        mat = run_dense_st(
            cfg["hf_id"], cfg["name"],
            class_texts, cqs_full,
            query_prefix=cfg["query_prefix"],
        )
        logger.info("  %s done in %.1fs", cfg["name"], time.time() - t0)
        all_results[cfg["name"]] = mat

    # ── Group 2: TF-IDF ───────────────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  GROUP 2 — TF-IDF Keyword Baseline")
    logger.info("═" * 65)
    t0 = time.time()
    all_results["TF-IDF"] = run_tfidf(class_texts, cqs_clean)
    logger.info("  TF-IDF done in %.1fs", time.time() - t0)

    # ── Group 3: SPLADE ───────────────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  GROUP 3 — SPLADE Sparse Neural Retrieval")
    logger.info("═" * 65)
    t0 = time.time()
    all_results["SPLADE"] = run_splade(class_texts, cqs_full)
    logger.info("  SPLADE done in %.1fs", time.time() - t0)

    # ── Group 4: ColBERT ──────────────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  GROUP 4 — ColBERT Late Interaction")
    logger.info("═" * 65)
    t0 = time.time()
    all_results["ColBERT"] = run_colbert(class_texts, cqs_full)
    logger.info("  ColBERT done in %.1fs", time.time() - t0)

    # ── Group 5: BioWordVec ───────────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  GROUP 5 — BioWordVec Domain-specific Embeddings")
    logger.info("═" * 65)
    t0 = time.time()
    all_results["BioWordVec"] = run_biowordvec(bio_path, class_texts, cqs_clean)
    logger.info("  BioWordVec done in %.1fs", time.time() - t0)

    # ── Save outputs ──────────────────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  SAVING RESULTS")
    logger.info("═" * 65)
    df_long = build_outputs(all_results, classes, cqs_raw, cqs_clean)

    # ── Print sample ──────────────────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("  SAMPLE — Top 5 Classes for first 3 CQs per model")
    logger.info("═" * 65)
    print_sample(df_long, classes, cqs_raw, n_cqs=3)

    logger.info("\n  ✓ Done!")
    logger.info("  → %s  (full ranked scores per CQ per model)", OUTPUT_XLSX)
    logger.info("  → %s  (long format, filterable)", OUTPUT_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enslaved ontology class retrieval evaluation")
    parser.add_argument("--classes",    default=CLASSES_FILE)
    parser.add_argument("--cqs",        default=CQS_FILE)
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip Qwen3-8B and Nemotron-8B")
    parser.add_argument("--biowordvec", default=BIOWORDVEC_PATH,
                        help="Path to BioWordVec .bin file")
    args = parser.parse_args()

    main(
        classes_file = args.classes,
        cqs_file     = args.cqs,
        skip_large   = args.skip_large,
        bio_path     = args.biowordvec,
    )