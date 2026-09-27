# Business Entity Resolution Pipeline

This folder contains a compact, reproducible pipeline for the business entity resolution task.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduction

Run the pipeline from this folder:

```bash
python src/run_pipeline.py --data-dir /path/to/dataset --output-dir ../../output
```

The script will:
1. read the train/test TSV files with the required tab separator,
2. build candidate pairs with language-agnostic blocking,
3. score candidates with a LightGBM model,
4. write the final outputs:
   - output/matching_results.tsv
   - output/candidate_pairs.tsv

## Required output

The final files must satisfy the challenge validator rules, including one row per Source 1 entity and candidate IDs being the exact set fed to the matcher.
