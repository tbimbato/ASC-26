# INSTRUCTIONS

## Environment

Python 3.10+.

```bash
pip install -r requirements.txt
```

Key deps: `pyroomacoustics` (simulation + RT60), `soundfile`, `librosa`,
`scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`.

---

## Phase 2 pipeline (current)

Train on synthetic RIRs, test on real ones (sim-to-real). Full run, in order:

```bash
# 1. generate the synthetic dataset (long, run yourself; --n-per-type sets size)
python src/simulate.py                 # -> data/sim/wav/*.wav + data/sim/manifest.csv

# 2. features for the synthetic set
python src/build_dataset.py --manifest data/sim/manifest.csv --out data/sim/features.csv

# 3. assemble the real held-out test set (BUT + AIR-binaural + ACE-Single)
python src/real_test.py                # -> data/real/manifest.csv  (--per-room caps RIRs/room)

# 4. features for the real set
python src/build_dataset.py --manifest data/real/manifest.csv --out data/real/features.csv

# 5. benchmark: in-sim CV + sim-to-real, classical models
python src/train.py                    # reads data/sim + data/real by default -> results/
```

`train.py` takes `--sim-features` / `--real-features` if you need non-default paths.
Any change to `room_types.py` needs a re-run from step 1; any change to feature
extraction (`utils.py`) needs a re-run of every `build_dataset.py` (steps 2 and 4).

Real datasets live in `data/raw/` (gitignored). `real_test.py` maps each dataset's
filenames to the taxonomy and pools RIRs by physical room. MIT survey and the
AIR/ACE bandlimited or multichannel captures are deliberately excluded, see
`notes.md` and the `real_test.py` docstring for why.

### Manifest format
Any CSV with a `path` column (to a WAV RIR) plus label columns carried through:
`room_id`, `label` (room-type name), `geometry`. `build_dataset.py` adds the
6 features to each row.

### Reading results
`train.py` writes `results/metrics.csv` and confusion matrices for two evals:
- `insim_5fold`: stratified 5-fold on the synthetic set (all 11 classes). Already
  room-independent (one RIR per simulated room), so this is a clean in-sim number.
- `sim2real`: train on synthetic (overlap classes only), test on the real held-out
  rooms. The honest generalization number the project is about.
- `sim2real_coarse`: same models, same predictions, labels collapsed to acoustic
  archetypes (office + meeting_room -> small_furnished) after prediction. The
  fine-vs-coarse gap measures how much error is intra-archetype. See the
  "half functional, half acoustic" note in `notes.md`/`AGENTS.md`.
- Report **per-class** metrics: the extreme classes (tank, forest, cathedral)
  are trivially separable and inflate the global score.

---

## Phase 1 (done, archived)

Real-RIR classification on BUT ReverbDB. Findings and confusion matrices are in
`results/phase1_but/`, the reasoning in `notes.md`. The phase-1 indexing script
was removed in the pivot; phase 1 is kept as a documented result, not a runnable
step.
