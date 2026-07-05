# ASC-26 — Acoustic Space Classification

Classifying rooms from their acoustic fingerprint. Given a measured room impulse response (RIR), can lightweight classical ML models recognize the type of space (office, lecture room, staircase...) using a handful of interpretable room-acoustic parameters?

Work in progress. MSc side project, Uppsala University (Machine Learning & Statistics).

## Idea

Every room colors sound in its own way: reverberation time, clarity, how fast energy decays. Instead of feeding raw audio to a large neural network, this project extracts a small set of standard room-acoustic descriptors from each RIR and benchmarks classical classifiers on them. The question: how far can interpretable, computationally cheap features go for acoustic space recognition, compared to neural baselines?

The angle comes from my background: I worked as an architect before moving into ML, so the features here (RT60, C80, D50, Ts) are the same quantities used in architectural acoustics practice.

## Data

[BUT Speech@FIT Reverb Database](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database): real RIRs measured in 9 rooms (offices, lecture rooms, meeting rooms, a staircase, a hotel room...). Not included in the repo; download separately into `data/raw/`.

## Pipeline

1. `src/extract_features.py` — walks the raw dataset, indexes RIR files into `data/processed/rir_paths.csv`
2. `src/build_dataset.py` — computes features per RIR → `data/processed/features.csv`
3. `src/utils.py` — onset-aligned feature extraction (RT60, EDT, C80, D50, Ts, DRR) and room→label mapping
4. `src/train.py` — benchmark over RandomForest, SVM, kNN, XGBoost with two evaluations (see below)

## Evaluation design

RIRs from the same room are highly correlated, so a naive split leaks room identity. The benchmark therefore reports two numbers:

- **A. Stratified 5-fold** (all 5 classes): room-dependent, an optimistic upper bound.
- **B. Leave-one-room-out** (classes covered by 2+ rooms: office, lecture room): the model never sees the test room. This is the honest generalization figure.

## Current results (1655 RIRs, 6 features)

| Eval | Best model | Accuracy | F1 macro |
|------|-----------|----------|----------|
| A. Stratified 5-fold (5 classes) | XGBoost | 0.93 | 0.89 |
| B. Leave-one-room-out (2 classes, 5 rooms) | RandomForest | 0.98 | 0.98 |

Six interpretable acoustic parameters are enough to recognize the type of space with high accuracy, and (on the classes where it can be tested) they transfer across rooms never seen in training. Confusion matrices and full metrics in `results/`.

## Status / roadmap

- [x] Dataset indexing and feature extraction
- [x] Onset-aligned energy parameters + EDT and DRR
- [x] Dual evaluation: stratified 5-fold + leave-one-room-out
- [x] Metrics and confusion matrices saved to `results/`
- [ ] Neural baseline for comparison (small CNN on raw/spectral RIR)
- [ ] Octave-band features (spectral RT60)
- [ ] Short writeup of findings

## Setup

```bash
pip install -r requirements.txt
python src/extract_features.py
python src/build_dataset.py
python src/train.py
```
