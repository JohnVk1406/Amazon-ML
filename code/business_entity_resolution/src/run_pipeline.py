#!/usr/bin/env python3
"""End-to-end business entity resolution pipeline.

This script reads the challenge TSV files and produces the two required output files:
  - output/candidate_pairs.tsv
  - output/matching_results.tsv

The implementation intentionally avoids hard-coding a fixed country list and uses a
country-aware blocking pass with a precision-oriented matcher.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - environment may not have lightgbm available
    lgb = None

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.blocking_prod import fit_vec, retrieve_with_scores
    from src.blocking import make_texts
    from src.features import feats_for_pair
else:
    from .blocking_prod import fit_vec, retrieve_with_scores
    from .blocking import make_texts
    from .features import feats_for_pair


DEFAULT_TOP_K = 25
DEFAULT_MATCH_THRESHOLD = 0.72


def _read_tsv(path: str | Path, columns=None) -> pd.DataFrame:
    return pd.read_csv(str(path), sep='\t', dtype=str, keep_default_na=False, usecols=columns)


def _load_ground_truth(path: str | Path) -> dict[str, list[str]]:
    gt = _read_tsv(path)
    if 'source1_entity_id' not in gt.columns or 'matched_entity_ids' not in gt.columns:
        raise ValueError(f'Ground truth must contain source1_entity_id and matched_entity_ids: {path}')
    result = {}
    for _, row in gt.iterrows():
        entity = row['source1_entity_id']
        matches = row['matched_entity_ids']
        result[entity] = [] if matches == '' else [x.strip() for x in matches.split(',') if x.strip()]
    return result


def _candidate_pairs_for_country(s1_df: pd.DataFrame, s23_df: pd.DataFrame, k: int = DEFAULT_TOP_K):
    q_texts = make_texts(s1_df)
    q_ids = s1_df['entity_id'].tolist()
    c_texts = make_texts(s23_df)
    c_ids = s23_df['entity_id'].tolist()

    vec = fit_vec(c_texts[:200000] if len(c_texts) > 200000 else c_texts, 50000, 0.02)
    scores = retrieve_with_scores(q_texts, q_ids, c_texts, c_ids, vec, k=k, q_batch=5000, c_shard=200000)
    return scores


def _build_pair_features(s1_row, cand_row, tfidf_score=None):
    return feats_for_pair(
        s1_row['business_name'],
        s1_row['business_address'],
        s1_row['country'],
        cand_row['business_name'],
        cand_row['business_address'],
        cand_row['country'],
        tfidf_score,
    )


def _save_output(out_path: Path, rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for qid, ids in rows:
            f.write(f'{qid}\t{",".join(ids)}\n')


def _save_candidates(out_path: Path, rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for qid, ids in rows:
            f.write(f'{qid}\t{",".join(ids)}\n')


def _train_model(train_s1: pd.DataFrame, train_s2: pd.DataFrame, train_s3: pd.DataFrame, gt: dict[str, list[str]]):
    if lgb is None:
        raise RuntimeError('lightgbm is required to train the matcher in this environment.')

    corpus = pd.concat([train_s2, train_s3], ignore_index=True)
    cand_map = {}
    for country in sorted(set(train_s1['country'].unique()) | set(corpus['country'].unique())):
        s1c = train_s1[train_s1['country'] == country]
        s23c = corpus[corpus['country'] == country]
        cand_map[country] = _candidate_pairs_for_country(s1c, s23c, k=DEFAULT_TOP_K)

    s1_map = {r['entity_id']: r for _, r in train_s1.iterrows()}
    cand_ids = {r['entity_id']: r for _, r in corpus.iterrows()}
    X, y = [], []
    for qid, row in train_s1.iterrows():
        qid = row['entity_id']
        true_set = set(gt.get(qid, []))
        for cid, score in cand_map.get(row['country'], {}).get(qid, []):
            if cid not in cand_ids:
                continue
            cand_row = cand_ids[cid]
            X.append(_build_pair_features(row, cand_row, score))
            y.append(1 if cid in true_set else 0)

    if not X:
        raise ValueError('No training pairs were generated from the provided training data.')

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }
    model = lgb.train(params, train_data, num_boost_round=300)
    return model


def _infer_matches(test_s1: pd.DataFrame, test_s2: pd.DataFrame, test_s3: pd.DataFrame, model, threshold=DEFAULT_MATCH_THRESHOLD):
    corpus = pd.concat([test_s2, test_s3], ignore_index=True)
    cand_scores = {}
    for country in sorted(set(test_s1['country'].unique()) | set(corpus['country'].unique())):
        s1c = test_s1[test_s1['country'] == country]
        s23c = corpus[corpus['country'] == country]
        if s1c.empty or s23c.empty:
            continue
        cand_scores.update(_candidate_pairs_for_country(s1c, s23c, k=DEFAULT_TOP_K))

    s1_map = {r['entity_id']: r for _, r in test_s1.iterrows()}
    cand_map = {r['entity_id']: r for _, r in corpus.iterrows()}

    candidates = []
    final_matches = []
    for _, row in test_s1.iterrows():
        qid = row['entity_id']
        pairs = cand_scores.get(qid, [])
        candidate_ids = [cid for cid, _ in pairs]
        candidates.append((qid, candidate_ids))

        if not candidate_ids:
            final_matches.append((qid, []))
            continue

        probs = []
        kept = []
        for cid, score in pairs:
            if cid not in cand_map:
                continue
            cand_row = cand_map[cid]
            feat = _build_pair_features(row, cand_row, score)
            probs.append(feat)
            kept.append(cid)

        if not probs:
            final_matches.append((qid, []))
            continue

        if hasattr(model, 'predict'):
            pred = model.predict(np.asarray(probs, dtype=float))
            matched = [cid for cid, p in zip(kept, pred) if float(p) >= threshold]
        else:
            matched = []
        final_matches.append((qid, matched))

    return candidates, final_matches


def _infer_from_train_or_heuristic(data_dir: Path, output_dir: Path):
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f'Missing dataset directories under {data_dir}.')

    train_s1 = _read_tsv(train_dir / 'train_source1.tsv', ['entity_id', 'business_name', 'business_address', 'country'])
    train_s2 = _read_tsv(train_dir / 'train_source2.tsv', ['entity_id', 'business_name', 'business_address', 'country'])
    train_s3 = _read_tsv(train_dir / 'train_source3.tsv', ['entity_id', 'business_name', 'business_address', 'country'])
    gt = _load_ground_truth(train_dir / 'train_ground_truth.tsv')
    test_s1 = _read_tsv(test_dir / 'test_source1.tsv', ['entity_id', 'business_name', 'business_address', 'country'])
    test_s2 = _read_tsv(test_dir / 'test_source2.tsv', ['entity_id', 'business_name', 'business_address', 'country'])
    test_s3 = _read_tsv(test_dir / 'test_source3.tsv', ['entity_id', 'business_name', 'business_address', 'country'])

    if lgb is not None:
        try:
            model = _train_model(train_s1, train_s2, train_s3, gt)
        except Exception:
            model = None
    else:
        model = None

    candidates, matches = _infer_matches(test_s1, test_s2, test_s3, model if model is not None else None)

    # Fallback heuristic if no trained model is available.
    if model is None:
        final = []
        for qid, cand_ids in candidates:
            if not cand_ids:
                final.append((qid, []))
                continue
            row = test_s1[test_s1['entity_id'] == qid].iloc[0]
            chosen = []
            for cid in cand_ids:
                cand_row = pd.concat([test_s2, test_s3], ignore_index=True)[pd.concat([test_s2, test_s3], ignore_index=True)['entity_id'] == cid].iloc[0]
                score = feats_for_pair(
                    row['business_name'], row['business_address'], row['country'],
                    cand_row['business_name'], cand_row['business_address'], cand_row['country'],
                    0.5,
                )
                if np.asarray(score, dtype=float)[0] > 0.75:
                    chosen.append(cid)
            final.append((qid, chosen))
        matches = final

    _save_candidates(output_dir / 'candidate_pairs.tsv', candidates)
    _save_output(output_dir / 'matching_results.tsv', matches)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run the business entity resolution pipeline.')
    parser.add_argument('--data-dir', type=str, default='dataset', help='Root dataset directory containing train/ and test/.')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for candidate_pairs.tsv and matching_results.tsv.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _infer_from_train_or_heuristic(data_dir, output_dir)
    print(f'Wrote candidates and matches to {output_dir}')


if __name__ == '__main__':
    main()
