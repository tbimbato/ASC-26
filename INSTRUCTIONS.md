# INSTRUCTIONS

## Environment

Python 3.10+.

```bash
pip install -r requirements.txt
```

Key deps: `pyroomacoustics` (simulation + RT60), `soundfile`, `librosa`,
`scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`.


## Phase 2 pipeline

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

# 6. neural baselines: same protocol, 4 nets (CNN at 3 capacities + pretrained ResNet18)
python src/nn/train_nn.py              # full sweep, hours; name models for a subset,
                                       # e.g. python src/nn/train_nn.py CNN-24k CNN-94k

# 7. paired tests between the two arms, needs 5 and 6 to have run
python src/significance.py             # -> McNemar at room and RIR level

# 8. control: feature arm re-run on the signal the network sees
python src/control_truncation.py       # -> results/control_truncation.csv

# 9. the README figure (needs the BUT corpus present)
python src/figures.py                  # -> results/sim_vs_real.png
```

`train_nn.py` merges into `results/metrics_nn.csv` per (model, seed): re-running a pair
replaces only its rows, and a pair already present is skipped, so an interrupted sweep
resumes. ResNet18 downloads its ImageNet weights once (~45 MB, cached by torchvision).

`train.py` takes `--sim-features` / `--real-features` if you need non-default paths.
Any change to `room_types.py` needs a re-run from step 1; any change to feature
extraction (`utils.py`) needs a re-run of every `build_dataset.py` (steps 2 and 4).

Real datasets are downloaded by hand into `data/raw/` (gitignored), and `real_test.py`
expects these folder names: `BUT_ReverbDB_rel_19_06_RIR-Only/`, `AIR_1_4/AIR_wav_files/`,
and any directory starting with `ACE` containing a `Single/` subtree. It exits with an
error if one is missing. It maps each corpus's filenames to the taxonomy and pools RIRs
by physical room. The MIT survey and the AIR/ACE bandlimited or multichannel captures
are excluded on purpose; the `real_test.py` docstring says why.

### Manifest format
Any CSV with a `path` column (to a WAV RIR) plus label columns carried through:
`room_id`, `label` (room-type name), `geometry`. `build_dataset.py` adds the
6 features to each row.

### Reading results
`train.py` writes `results/metrics.csv`, per-RIR predictions on the real set in
`results/preds.csv`, and confusion matrices. The evals are:
- `insim_5fold`: stratified 5-fold on the synthetic set (all 10 classes). Already
  room-independent (one RIR per simulated room), so this is a clean in-sim number.
- `sim2real`: train on synthetic (overlap classes only), test on the real held-out
  rooms. This is the generalization number the project is about.
- `sim2real_coarse`: same models, same predictions, labels collapsed to acoustic
  archetypes (office + meeting_room -> small_furnished) after prediction. The
  fine-vs-coarse gap measures how much error is intra-archetype.
- `insim_overlap_5fold`: in-sim on the 4 overlap classes only, so the in-sim to
  sim2real drop is computed on the same task for both model families.
- `abl_rt60_only` and `abl_no_<feature>`: leave-one-feature-out and RT60-alone, fine
  and coarse. These are most of the rows in `metrics.csv`.

The in-sim 10-class numbers are inflated by the extreme classes (cathedral, forest,
patio), which are trivially separable. Read the confusion matrices rather than the
global score; per-class tables were never produced.

`train_nn.py` writes `results/metrics_nn.csv` with the `nn_` prefixed counterparts
(`nn_insim`, `nn_insim_overlap`, `nn_sim2real[_coarse][_adabn]`) plus per-sample
predictions on the real set in `results/preds_nn.csv`. Rows scored on the real set
also carry `baseline`, `ci_low`, `ci_high` and `acc_rooms`.


## Phase 1 (done, archived)

Real-RIR classification on BUT ReverbDB. Findings and confusion matrices are in
`results/phase1_but/`. The phase-1 indexing script
was removed in the pivot; phase 1 is kept as a documented result, not a runnable
step.

## archive/

`archive/exploration_phase1.ipynb` is phase-1 material kept for the record. It reads
paths that no current script writes and does not run against this pipeline.
