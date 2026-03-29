# Embeddings Evaluation

This repository evaluates embeddings for the NEN schema by converting ontology triples into natural language sentences, embedding them with different models, and assessing retrieval quality using competency questions.

## Project Goals

- Convert NEN schema triples (`Subject`, `Predicate`, `Object`) into natural language statements
- Embed schema triples and competency questions with multiple embedding models
- Evaluate retrieval quality using:
  - Cosine similarity
  - Hit@K (K = 1, 3, 5)
  - Mean Reciprocal Rank (MRR)
  - Average cosine similarity for relevant triples

## Files

- `nen_embedding_eval.py` - main evaluation script
- `NEN.csv` - source schema triples
- `nen_eval_results.csv` - evaluation output
- `nen_embedding_eval.log` - runtime logs

## Requirements

Install the Python dependencies before running the script:

```bash
pip install torch transformers sentence-transformers pandas numpy scikit-learn tabulate tqdm
```

If you want to use quantized Hugging Face models, also install:

```bash
pip install bitsandbytes accelerate
```

## Usage

Run the evaluation script from the repository root:

```bash
python nen_embedding_eval.py
```

The script will:

1. Load `NEN.csv`
2. Convert triples into text sentences
3. Embed schema triples and competency questions
4. Compute evaluation metrics
5. Save results to `nen_eval_results.csv`
6. Log runtime output to `nen_embedding_eval.log`

## Models Included

The script currently evaluates the following embedding models:

- `BAAI/bge-large-en-v1.5`
- `Qwen/Qwen3-Embedding-8B`
- `nvidia/llama-embed-nemotron-8b`

## Notes

- Large models such as Qwen3 and Nemotron may require significant RAM/VRAM.
- The script uses Python logging for output and writes logs to `nen_embedding_eval.log`.
- You can change the CSV path by editing `NEN_CSV_PATH` in `nen_embedding_eval.py`.

## License

No license is specified.
