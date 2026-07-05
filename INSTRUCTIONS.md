# INSTRUCTIONS

## Environment

Python 3.10+.

```bash
pip install -r requirements.txt
```

Key deps: `pyroomacoustics` (simulation + RT60), `soundfile`, `librosa`,
`scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`.

---

## Phase 2 pipeline (current, under construction)

Train on synthetic RIRs, test on real ones (sim-to-real).

```bash
# 1. generate the synthetic dataset  (simulate.py: skeleton, not runnable yet)
python src/simulate.py                 # -> data/sim/wav/*.wav + data/sim/manifest.csv

# 2. compute features from any manifest (works now)
python src/build_dataset.py --manifest data/sim/manifest.csv --out data/sim/features.csv

# 3. benchmark classical models
python src/train.py --features data/sim/features.csv   # -> results/
```

Before step 1 works, validate the room-type taxonomy, geometries and ranges in
`src/room_types.py`, then implement the builders in `src/simulate.py`.

The real test set (BUT, MIT survey, OpenAIR) will be assembled by
`src/real_test.py` into a manifest with the same columns, then fed through the
same steps 2-3. Not built yet.

### Manifest format
Any CSV with a `path` column (to a WAV RIR) plus label columns carried through:
`room_id`, `label` (room-type name), `geometry`. `build_dataset.py` adds the
6 features to each row.

### Reading results
- `stratified5fold`: rooms can appear in train and test. Optimistic upper bound.
- `leave1roomout`: the test room is never seen in training. The honest number.
- Report **per-class** metrics: the extreme classes (tank, forest, cathedral)
  are trivially separable and inflate the global score.

---

## Phase 1 (done, archived)

Real-RIR classification on BUT ReverbDB. Findings and confusion matrices are in
`results/phase1_but/`, the reasoning in `notes.md`. The phase-1 indexing script
was removed in the pivot; phase 1 is kept as a documented result, not a runnable
step.
