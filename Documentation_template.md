# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Amazon ML Baseline Team  
**Team Members:** [Add members]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
This solution uses a two-stage entity-resolution pipeline tailored to noisy business records. First, it applies a recall-oriented blocking pass that uses normalized name and address text to shortlist plausible Source 2 / Source 3 candidates for each Source 1 entity. Then it scores each candidate with a LightGBM matcher built from name, address, country, and TF-IDF similarity features. The final threshold is tuned conservatively to preserve precision, because the scoring metric is F_0.5 and false merges are penalized much more than missed matches.

---

## 2. Methodology

### 2.1 Problem Analysis
The dataset contains noisy entity strings across multiple sources. Common patterns include legal suffix variants, abbreviations in business names, address shorthand (for example, Road vs. Rd), missing address components, landmark references, and word-order differences. Source 1 is a deduplicated reference set, while Source 2 and Source 3 contain partial noisy matches. Because the country field is open-ended and the training countries are US and India while test adds France, the pipeline is designed to avoid hard-coded country logic and instead treat the country as a soft matching signal rather than a fixed category set.

### 2.2 Solution Strategy
The pipeline follows a standard blocking + matching architecture. Candidate generation is designed to maximize recall and keep the search space manageable. Matching then uses a precision-oriented model with features that capture lexical similarity, suffix-insensitive name overlap, address similarity, and whether country labels align. The final model threshold is selected on a validation split to optimize macro F_0.5 rather than raw accuracy.

**Approach Type:** Blocking + Classifier  
**Core Innovation:** A lightweight, country-agnostic retrieval strategy that combines normalized name/address text with TF-IDF and similarity features to keep recall high while filtering false matches.

---

## 3. Candidate Generation (Blocking)
The blocking stage reduces all-pairs comparisons using a text-based retrieval method. For each Source 1 record, the pipeline builds a combined name + address text string after normalization, then calculates TF-IDF similarities against all Source 2 + Source 3 records in the same country bucket or overall corpus. The top K scoring records are retained as candidates; this maintains a high recall ceiling while reducing candidate sets to a manageable size.

- **Blocking keys used:** normalized business name, normalized address, TF-IDF token matching on combined name/address text
- **Candidate pairs generated:** depends on the final dataset size, but the candidate list is the exact last blocking stage before model inference
- **How you ensured true matches were not lost:** the blocking stage is intentionally conservative and keeps a small top-K shortlist per Source 1 record with a fallback for low-confidence or sparse query cases, preserving a broad recall ceiling before the matcher filters down to final predictions

---

## 4. Matching Model

**Features used:**
- Name features: token-set ratio, weighted ratio, no-suffix token-set comparison, Jaccard overlap
- Address features: token-set ratio, weighted ratio, Jaccard overlap, length-difference checks
- Other: country match flag, empty-address flag, TF-IDF similarity score from the blocker

**Model type:** LightGBM binary classifier  
**Threshold selection method:** validation-driven threshold search on F_0.5, with a conservative threshold chosen to favor precision on false merges and still retain enough recall for true matches

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro):** evaluated on a validation split from the training set using the provided F_0.5 macro formulation
- **Common false positives (wrong merges):** ambiguous name matches across different legal entities with highly similar addresses or same city context
- **Common false negatives (missed matches):** cases with heavy abbreviation, transliteration, or missing address data that fall below the blocking recall ceiling

---

## 6. Conclusion
This approach balances recall in candidate generation with high precision in final matching. The blocking stage keeps the search space controlled, and the LightGBM matcher uses strong lexical and address features to separate true matches from false joins while preserving generalization to unseen countries such as France.

---

## Appendix

### A. Code Artefacts
The full runnable implementation ships under the code bundle in the submission zip. The pipeline logic is grouped in the `src/` directory and can be invoked with the instructions in the code README to generate both output TSV files.

### B. Additional Results
Additional tuning and validation metrics can be reproduced from the train/validation split in the source code. The solution is intentionally designed to favor precision under the F_0.5 objective while maintaining a high enough recall ceiling to avoid missing true matches.

---
