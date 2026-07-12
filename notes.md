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
