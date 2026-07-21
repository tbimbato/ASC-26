# Project diary

## First phase: real RIRs, BUT ReverbDB [March '26 - June '26]

Tried to classify room type from real RIRs (`BUT_ReverbDB_rel_19_06_RIR-Only`). Built the classical pipeline:
- 6 interpretable room-acoustic features, onset-aligned: RT60, EDT, C80, D50, Ts, DRR.
- Classifiers: RandomForest, SVM, kNN, XGBoost.
- Two evaluations: stratified 5-fold (optimistic, room can leak) and leave-one-room-out (honest, unseen room).

Problems found:
- The dataset has only **9 rooms** (mapping to ~5 room-type labels), with thousands of RIR measurements per room. The unit for this task is the **room**, not the measurement, so the effective sample size is tiny. Way too few distinct rooms to train a neural network fairly, or to claim generalization.
- Proof of the ceiling: leave-one-room-out only runs on 2 classes (office, lecture_room), the only ones with 2+ rooms. The other classes have a single room each, so room-independent generalization can't even be tested.
- Merging more real datasets does not fix it cleanly: BUT is few-rooms / many-recordings, while the datasets that add room diversity (e.g. MIT survey) are many-rooms / one RIR each. Merging at measurement level is heterogeneous, BUT dominates, and different capture setups risk a dataset-bias leak (the model learns "which dataset", not "which room type").

Conclusion: more ROWS do not help, only more ROOMS do. Instead of chasing scarce, heterogeneous real datasets, generate a synthetic dataset whose dimensionality fits both classical ML and a neural network.

## Second phase: synthetic dataset + sim-to-real [July '26]

Plan:
- Generate a synthetic, balanced, labelled RIR dataset with **pyroomacoustics** (control room type, dimensions, materials, source/mic placement). Unlimited rooms, perfect labels, no dataset bias.
- Train on synthetic. Keep the REAL datasets (BUT, MIT survey, OpenAIR) as a **held-out test set**. This is a sim-to-real evaluation: learn in simulation, prove it on real recordings.
- Reuse the exact same 6 features and classifiers, then add the neural / SOTA comparison.

Research question: **with the same synthetic training set, do interpretable classical features generalize to real rooms as well as (or better than) a neural network?** The hypothesis: physics-based features transfer sim-to-real better because they capture room physics, while a NN can overfit to simulator artifacts.

### Room-type taxonomy

I defined the set of room types to simulate. I split them into two groups.

Everyday rooms, where the hard discrimination lives: small office, meeting room, lecture hall, corridor, stairwell, large hall, bathroom. These are acoustically close to each other and are the real challenge.

Extreme / exotic spaces, added on purpose: cathedral, gas tank, outdoor patio, forest. I add them for three reasons. They widen the acoustic range of the dataset, they make the problem more interesting than a set of boxes, and they map to spaces that actually exist in the real held-out datasets (OpenAIR has real cathedrals, tanks, tunnels and outdoor recordings), so they stay testable sim-to-real rather than being pure fantasy.

### Geometry: not just rectangular boxes

At first my taxonomy only described rectangular boxes (shoebox: width, depth, height). That is fine for an office, but it is wrong for a round tank, for a cathedral, for a patio open on one side, or for a forest with no walls at all. So I needed a geometry abstraction.

pyroomacoustics is not limited to boxes. It can build an arbitrary polygonal floor plan extruded to a height, with a different material on each wall. On top of that, a wall set to near-total absorption behaves like open air (energy hitting it is gone). From these primitives I get all the cases I need:

- **shoebox**: the everyday rectangular rooms.
- **polygon** (extruded floor plan): L-shapes and the cathedral. Curved surfaces (vaults, apses) are approximated with flat facets.
- **cylinder**: the gas tank, built as an n-gon prism (e.g. 24 sides) to approximate the round wall.
- **partial** enclosure: the patio, a box with one or two walls set to near-total absorption to act as open air, plus a reflective ground.
- **open_field**: the forest / outdoor. pyroomacoustics simulates enclosed spaces, so true free field is out of scope. I approximate it as a very large, almost fully absorbing box plus a reflective ground and high scattering. Acoustically this gives mostly the direct sound and a ground reflection with almost no reverberation, which is close to how open space behaves.

In code this is a `geometry` field per room type and one builder per geometry kind in `simulate.py`.

### Honesty / limitations (to state in the writeup)

- The exotic classes are acoustically extreme and trivially separable: RT60 alone tells a tank (long metallic tail) from a forest (almost none). They inflate the overall accuracy. I will report **per-class metrics** and keep the focus on the hard discrimination among the everyday rooms, not on a single global number.
- The exotic geometries are approximations. Curvature is faceted, so focusing effects are not captured, and the outdoor case is a large absorbing box, not a true free field. These are modelling choices, stated openly.
- Why I still expect the interpretable features to do well: RT60, EDT, C80, D50, Ts and DRR are physical descriptors of the room, relatively robust to the capture setup. That is exactly the property I bet on for sim-to-real transfer, against a neural network that could latch onto simulator-specific artifacts.

### Pipeline (target)
1. `src/room_types.py` — room-type taxonomy + geometry + parameter distributions (DRAFT, to validate)
2. `src/simulate.py` — sample rooms, dispatch by geometry, generate RIRs, write `data/sim/manifest.csv`
3. `src/build_dataset.py` — features from a manifest (sim or real) -> features CSV
4. `src/train.py` — in-sim eval + sim-to-real eval, classical vs neural
5. `src/real_test.py` — assemble + relabel the real test set (BUT/MIT/OpenAIR) (later)

### Open decisions
- Validate the room-type taxonomy, geometry choices and dimension/absorption ranges (`room_types.py`), especially the exotic spaces where my estimates are rough.
- Neural side (phase 3): small CNN on RIR spectrograms, or convolve RIRs with speech and use a pretrained SOTA audio model. Decide when we get there.

### Fixing the leave-one-room-out confusion (July '26)

Ported `train.py` from phase 1 as-is at first, including the leave-one-room-out CV. Ran it on the synthetic set and it hung for a long time on "Leave-one-room-out on classes [...11 classes...] (2200 rooms, 2200 samples)". Killed it.

The reason: in the synthetic set every `room_id` is a distinct simulated room generated once, so rooms == samples, 1:1. Leave-one-room-out degenerates into leave-one-sample-out CV, thousands of folds, no extra rigor over a plain stratified k-fold, just very slow (refits every model per fold).

Leave-one-room-out only ever made sense for the real BUT set, where a handful of physical rooms are each recorded multiple times (few rooms, many RIRs), so a naive split leaks the room identity. That axis doesn't exist in the synthetic set by construction.

Fixed `train.py`:
- Eval A stays a stratified 5-fold CV on the full synthetic set (`insim_5fold`). It's already room-independent because rooms don't repeat.
- Eval B is now the actual sim-to-real test (`sim2real`): train on synthetic restricted to the overlap classes (office, meeting_room, lecture_room, staircase), test on the real held-out set for those classes. This is the eval that matters for the paper's claim.

Old `cm_leave1roomout_*.png` / `cm_stratified5fold_*.png` in `results/` are stale (from the old scheme, one run partial from the kill). Will be replaced by `cm_insim_5fold_*.png` and `cm_sim2real_*.png` on the next run.

### Expanding the real test set (July '26)

First sim2real run: RF in-sim 74%, sim2real 37%. The gap looked like a systematic reverberation offset: staircase and lecture_room transfer well (sim medians ~ real), office and meeting_room fail because the sim looked drier than the real BUT rooms. But the real set was tiny (office 3 rooms, meeting_room/staircase 1 room each), so "the real median" often came from a single room. (NOTE: every absolute RT60 number in this section and the two above it is inflated x2, a bug found later the same day. See "RT60 bug" below. The x2 is a constant factor on one feature so the accuracies are unaffected, but the raw seconds here are wrong: e.g. the "BUT meeting_room at 3.0s" was really ~1.5s.)

Checked room_types.py against the acoustics literature (Sabine at midpoint dims vs published RT60 targets: BB93, ANSI/ASA S12.60, Long, Beranek). The sim values were already mostly in range. Widened absorption slightly for office/meeting_room/large_hall/bathroom to cover the untreated/hard end, justified by literature, NOT fitted to BUT (that would leak the test set). Re-running simulate after this.

Tried to add real rooms from MIT survey and AIR (Aachen), both dropped into data/raw:
- **MIT survey: rejected, but not because it's a bad dataset.** Traer & McDermott (PNAS 2016) recorded at 1.5m (conversational distance) in real everyday spaces with real background noise, for a perceptual study of how reverb sounds in daily life, not to extract T30-style RT60. Most files only have a short window of usable dynamic range before the ambient noise floor (median ~0.7s, some as low as 0.18s), which is fine for their purpose but not enough decay range for a standard RT60/C80/DRR estimate; feeding it through extract_features gives numbers like a 0.18s "large brick open-plan office", which is a sim artifact of the estimator meeting insufficient dynamic range, not a claim about the room. Good scope-mismatch finding to cite: physical decay features need a measurement designed for it, not every real recording of a room.
- **AIR: kept binaural only.** Full-band, full-length (3s) IRs for office, meeting, lecture, stairway = 4 clean rooms. The AIR *phone* recordings are telephone-band (300-3400 Hz), which inflates C80; excluded as a capture-bias confound. corridor is only in the .mat files; skipped (MIT would have covered it but MIT is out).

Real test set was BUT + AIR-binaural = 12 rooms (`real_test.py` rewritten).

### Adding ACE Challenge (July '26)

ACE official site was down; got it from the Zenodo mirror (search "Data Corpus for the IEEE-AASP Challenge on the Acoustic Characterization of Environments"). The corpus ships several mic configs (Single, Crucifix, EM32 Eigenmike, a linear 8-ch array, and mobile/Chromebook captures); only **Single** (single full-band studio mic) was downloaded, the rest are multichannel or bandlimited consumer devices (same capture-bias concern as AIR-phone).

ACE/Single is small by design (it's a characterisation benchmark, not a bulk corpus): 7 rooms x 2 mic/source positions = 14 RIR files, each ~300KB. Each position folder also has 3 long noise recordings (ambient/babble/fan, ~17MB each) for the challenge's denoising task; irrelevant here, only `*_RIR.wav` is used. Rooms: Office_1/2, Meeting_Room_1/2, Lecture_Room_1/2 (all map directly onto existing classes), and Building_Lobby (no clean mapping onto the taxonomy, e.g. not really corridor or large_hall, skipped for now).

Position 1 and 2 of the same room are the same physical room, not two rooms, so `collect_ace()` in `real_test.py` pools both under one room_id. Net add: 7 rooms, mostly on the classes that matter (office, meeting_room, lecture_room).

Sanity check on ace features: RT60 physically sensible per room (office ~0.7-0.8s, meeting_room ~0.85-0.92s, lecture_room 1.3-3.0s) and consistent across the two mic positions of the same room (e.g. office_1: 0.79 vs 0.72s), which is what a stable estimator should give. Have not yet cross-checked against ACE's own published ground-truth RT60/DRR annotations (need to locate/attach that metadata file separately from the audio download).

Real manifest is now BUT + AIR-binaural + ACE-Single = **18 rooms**, 72 RIRs (`data/real/manifest.csv`, features rebuilt into `data/real/features.csv`). Overlap classes: office 6 rooms, lecture_room 5, meeting_room 4, staircase 2 (ACE doesn't cover corridor/staircase/bathroom). Big jump from the original BUT-only real set (office 3, meeting_room/lecture_room/staircase 1 each).

### RT60 bug: the x2 (July '26)

Got ACE's own published ground-truth (`..._Corpus_Mean_DRRs_and_T60s.csv` and the per-measurement `..._test_t60_DRR_measurement_results.csv`, the T60/DRR the challenge organisers measured per room). Used it to validate extract_features, and it exposed a real bug in utils.py.

`utils.extract_features` had:
    rt60 = pra.measure_rt60(ir, fs=fs, decay_db=30) * 2
The `* 2` was wrong. Reading the pyroomacoustics source: `measure_rt60` already extrapolates the fitted decay slope to a full -60 dB internally (`_fit_exp_and_extrapolate(..., extrapolate_value_db=-60.0)`, hardcoded). `decay_db=30` only sets how much of the curve is used for the slope fit (a T30-style fit), the returned value is already the full RT60. Our `* 2` doubled a number that was already the final estimate. The "T30 -> T60 needs x2" reasoning was a mistake: that doubling is what happens *inside* pra, not on top of it.

Verified against ACE office room 502 (Single mic), fullband ground-truth T60 = 0.364 s (pos1) and 0.310 s (pos2):
  - with the `* 2` bug: 0.788 / 0.716 s  (about double, wrong)
  - fixed (no `* 2`):   0.394 / 0.358 s  (matches GT closely)
Fixed in utils.py, comment there records the ACE check.

Impact on results: none on the accuracies. The `* 2` is a constant factor on a single feature; StandardScaler (SVM/kNN) and tree splits (RF/XGB) are both invariant to it. Re-ran the whole pipeline after the fix and `results/metrics.csv` was byte-identical to before. What *was* wrong: every absolute RT60 value reported earlier today (sim/real medians, the Sabine-vs-literature comparison) was x2 too high. Corrected medians below.

Corrected RT60 medians (overlap classes, after fix):
    class          sim    real   real/sim
    office         0.36   0.61   1.69
    meeting_room   0.54   0.44   0.82
    lecture_room   0.85   1.06   1.25
    staircase      1.83   1.36   0.74
So there is no single "sim is always too dry" story anymore. Only office still shows a real gap (sim 0.36 vs real 0.61). meeting_room and staircase are now roughly matched or slightly reversed. The room_types.py literature calibration still stands (it was done with hand-computed Sabine, independent of the buggy pra call), it just needs its example numbers read at the correct scale.

### The taxonomy is half functional, half acoustic (July '26)

sim2real on the 4 overlap classes, corrected pipeline, bigger real set: RF/XGB acc ~0.40, f1 ~0.37. The XGBoost sim2real confusion matrix shows the whole loss is basically one class: office is 2/24 correct, with 12 real offices predicted lecture_room and 10 predicted meeting_room. Everything else (lecture_room 17/19, staircase 5/10) is decent for pure sim-to-real.

Dug into it, and this is the important finding of the project, not a bug to patch. The label set mixes two different axes:
  - Functional labels (what people do there): office, meeting_room. A small office and a small meeting room are the *same acoustic object*, a small furnished shoebox. The real meeting_room RT60 range alone spans 0.31-1.61 s: one functional name covers a closet to a hall. No acoustic feature can (or should) separate office from small-meeting, because acoustically they are not separate.
  - Acoustic labels (geometry/material give a signature): staircase, corridor, cathedral, gas_tank, bathroom. These have a real fingerprint and the features nail them.

RT60/C80/DRR describe the *space*, not its function. Where function and acoustics coincide (a stairwell is always tall+hard) classification is easy; where one function has variable acoustics (meeting_room) or two functions share acoustics (office ~ small meeting) the features refuse to separate them. That is the honest characteristic of physics-based features, and you can only *see* it because the features are interpretable (you can point at RT60 and explain every confusion; a NN making the same errors would be a black box).

Quantified it. Merging office+meeting_room into one a-priori acoustic archetype ("small_furnished"), keeping lecture_room and staircase:
    level               classes                                        acc    f1
    fine   (functional) office, meeting_room, lecture_room, staircase  0.40   0.37
    coarse (acoustic)   small_furnished, lecture_room, staircase       0.66   0.63
The +26 points is, literally measured, how much of the "error" was a functional distinction acoustics cannot make. The residual coarse error (14 small_furnished -> lecture_room) is the genuine small->medium size gradient, where a big meeting room really does become a small lecture room. The office calibration (sim too dry) is a minor separate contributor, de-prioritised: chasing it risks fitting the test set.

Decision: do NOT invent a new taxonomy from scratch (that throws away the architectural-room-type identity, the project's whole angle). Instead adopt a two-level evaluation, defined a-priori by acoustic archetype not by peeking at the confusion matrix: keep the functional labels, add a coarse acoustic grouping, report both. The gap between the two levels is a headline result.

### Project spine / where the neural baseline fits (July '26)

Restating the point after the taxonomy scare, because it briefly felt like the project was collapsing. It is not. The project was never "build the best room-type classifier" (weak portfolio). It is a controlled sim-to-real study: with the same synthetic training set, do interpretable physical features transfer to real rooms as well as, or more robustly than, a learned neural representation? The deliverable is the comparison + the interpretability, not a leaderboard number.

The task ceiling (office/meeting overlap) does NOT undermine the neural comparison, because it hits both models equally: both train on the same synthetic set, test on the same real held-out set, same metrics. The cleanest metric for the paper, immune to the task's intrinsic ambiguity, is the *drop from in-sim to sim2real* per model. The central hypothesis (a NN overfits simulator artifacts, physical features do not) is exactly: does the NN degrade more from synthetic to real than the 6 features do? That is a within-model relative measure, so the ambiguous ceiling cancels out.

Neural baseline plan, same protocol as the classical side:
  - train on synthetic RIRs, test on the real held-out set;
  - small CNN on the RIR (spectrogram or log energy-decay curve), decide input later;
  - compare on three axes: (1) sim2real accuracy at fine + coarse, (2) robustness = in-sim -> sim2real drop (the key axis), (3) interpretability (6 readable numbers vs black box).
Any outcome is publishable: features hold and NN collapses -> hypothesis confirmed; NN holds better -> we quantify the cost of interpretability. No losing outcome.

Honest caveats to state in the writeup: ~18 real rooms and 4 overlap classes is low statistical power for the NN-vs-features comparison; the fine everyday discrimination is genuinely hard; the exotic classes inflate any global accuracy and are reported separately.

Next steps: (1) wire the two-level eval into train.py (add a sim2real_coarse alongside the fine one, with the archetype map defined explicitly); (2) build the neural baseline on the same split; (3) optionally cross-check extract_features against the full ACE ground-truth table (not just the one office room spot-checked).

### Coarse eval wired into train.py (21 July '26)

Done next-step (1): `train.py` now reports `sim2real_coarse` next to the fine
eval. Implementation choice that matters for credibility: the coarse numbers
come from the *same models and same predictions* as the fine eval, with labels
collapsed through an explicit a-priori `COARSE_MAP` (office + meeting_room ->
small_furnished) only at scoring time. Nothing is merged before training, so
nobody can claim we tuned the training to the test set, and the fine-coarse
gap literally measures how much error is intra-archetype.

Official numbers (67 real samples, 17 rooms):
    model         fine acc  coarse acc
    RandomForest    0.403     0.701
    SVM             0.388     0.657
    kNN             0.358     0.627
    XGBoost         0.418     0.642
Slightly better than the hand-measured 0.66 from last week (that one merged
labels before training; re-scoring the fine model is the cleaner protocol and
happens to also score higher for RF). The ~+28 points fine->coarse is now an
official, reproducible result, not a side calculation.

Remaining: neural baseline on the same protocol, then the report.
