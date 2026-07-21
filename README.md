# ASC-26 — Acoustic Space Classification

Recognizing the *type* of a space (office, lecture hall, stairwell, cathedral, forest...) from its acoustic fingerprint, the room impulse response (RIR). The core question: how far do a handful of interpretable, physics-based room-acoustic parameters go, compared to a neural network, and which transfers better from simulation to real rooms?

Work in progress. MSc side project, Uppsala University (Machine Learning & Statistics).

## Two phases

**Phase 1 (done): real RIRs, BUT ReverbDB.**
Built the classical pipeline (6 interpretable features, 4 classifiers, honest leave-one-room-out evaluation). It works, but the dataset has only 9 rooms, so it is too small to train a neural network fairly or to claim generalization across room types. Findings and confusion matrices are archived in `results/phase1_but/`, the reasoning is in `notes.md`.

**Phase 2 (current): synthetic dataset + sim-to-real.**
Instead of chasing scarce, heterogeneous real datasets, generate a large, balanced, labelled RIR set with a room-acoustics simulator (`pyroomacoustics`): unlimited rooms, perfect labels, no dataset bias. Train on the synthetic set, then test on **real** held-out recordings (BUT ReverbDB + AIR-binaural + ACE Challenge, 18 rooms). The real question becomes sim-to-real transfer.

**Research question:** with the same synthetic training set, do interpretable classical features generalize to real rooms as well as (or better than) a neural network? Hypothesis: physical features transfer better because they describe room physics, while a NN can overfit to simulator artifacts. The clean metric is the drop from in-sim to sim-to-real per model, not raw accuracy.

**A finding that shapes the story:** the labels mix two axes. Some are functional (office, meeting room) and some are acoustic (stairwell, cathedral, tank). Acoustic features describe the space, not its function, so an office and a small meeting room are genuinely inseparable, while a stairwell or a cathedral is unmistakable. Results are reported at two levels, fine (functional) and coarse (acoustic archetype); the gap between them is part of the result. See `notes.md`.

## Room types

Everyday rooms (the hard discrimination): small office, meeting room, lecture hall, corridor, stairwell, large hall, bathroom.
Extreme spaces (wide acoustic range, in-simulation only, no real counterpart in the held-out set): cathedral, outdoor patio, forest.

These need more than rectangular boxes, so each type carries a `geometry` (shoebox, polygon, cylinder, partial enclosure, open field). See `src/room_types.py` and `notes.md` for the geometry design and its honest limitations.

## Pipeline

| Step | Script | Status |
|------|--------|--------|
| Room-type taxonomy + geometry | `src/room_types.py` | ready (literature-calibrated) |
| Generate synthetic RIRs | `src/simulate.py` | ready |
| Features from a manifest | `src/build_dataset.py` | ready (sim or real) |
| Feature extraction (6 params) | `src/utils.py` | ready (RT60 validated vs ACE) |
| Assemble real test set | `src/real_test.py` | ready (BUT + AIR + ACE) |
| Benchmark + evaluation | `src/train.py` | ready (in-sim + sim2real fine/coarse) |
| Neural baseline | todo | next |

Setup and step-by-step in [INSTRUCTIONS.md](INSTRUCTIONS.md). Project diary in [notes.md](notes.md).
