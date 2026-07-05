# ASC-26 — Acoustic Space Classification

Recognizing the *type* of a space (office, lecture hall, stairwell, cathedral, forest...) from its acoustic fingerprint, the room impulse response (RIR). The core question: how far do a handful of interpretable, physics-based room-acoustic parameters go, compared to a neural network, and which transfers better from simulation to real rooms?

Work in progress. MSc side project, Uppsala University (Machine Learning & Statistics).

## Two phases

**Phase 1 (done): real RIRs, BUT ReverbDB.**
Built the classical pipeline (6 interpretable features, 4 classifiers, honest leave-one-room-out evaluation). It works, but the dataset has only 9 rooms, so it is too small to train a neural network fairly or to claim generalization across room types. Findings and confusion matrices are archived in `results/phase1_but/`, the reasoning is in `notes.md`.

**Phase 2 (current): synthetic dataset + sim-to-real.**
Instead of chasing scarce, heterogeneous real datasets, generate a large, balanced, labelled RIR set with a room-acoustics simulator (`pyroomacoustics`): unlimited rooms, perfect labels, no dataset bias. Train on the synthetic set, then test on **real** held-out recordings (BUT, MIT survey, OpenAIR). The real question becomes sim-to-real transfer.

**Research question:** with the same synthetic training set, do interpretable classical features generalize to real rooms as well as (or better than) a neural network? Hypothesis: physical features transfer better because they describe room physics, while a NN can overfit to simulator artifacts.

## Room types

Everyday rooms (the hard discrimination): small office, meeting room, lecture hall, corridor, stairwell, large hall, bathroom.
Extreme spaces (wide acoustic range, and they exist in the real test sets): cathedral, gas tank, outdoor patio, forest.

These need more than rectangular boxes, so each type carries a `geometry` (shoebox, polygon, cylinder, partial enclosure, open field). See `src/room_types.py` and `notes.md` for the geometry design and its honest limitations.

## Pipeline

| Step | Script | Status |
|------|--------|--------|
| Room-type taxonomy + geometry | `src/room_types.py` | draft, to validate |
| Generate synthetic RIRs | `src/simulate.py` | skeleton |
| Features from a manifest | `src/build_dataset.py` | ready (sim or real) |
| Feature extraction (6 params) | `src/utils.py` | ready |
| Benchmark + evaluation | `src/train.py` | ready (classical) |
| Assemble real test set | `src/real_test.py` | later |

Setup and step-by-step in [INSTRUCTIONS.md](INSTRUCTIONS.md). Project diary in [notes.md](notes.md).
