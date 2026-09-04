# Project diary

## Phase 1 - real RIRs only (Mar-Jun '26)

BUT ReverbDB, 6 ISO 3382 features (RT60, EDT, C80, D50, Ts, DRR) + RF / SVM / kNN /
XGBoost. Abandoned: 8 usable rooms, thousands of RIRs each. The unit of analysis is the
room, not the measurement, so leave-one-room-out only ran on the 2 classes with more
than one room. More rows do not help, only more rooms do.

Merging other real corpora does not fix it: BUT is few-rooms/many-recordings, the
many-room corpora are one-RIR-each, and merging at measurement level lets the model
learn which dataset it is looking at.

## Phase 2 - train on simulation, test on real (Jul-Aug '26)

### Design

- 10 room types, 5 geometry builders (shoebox, polygon, cylinder, partial enclosure,
  open field); cylinder is unused since gas_tank was dropped, so the generated set
  spans four. Absorption ranges anchored to BB93, ANSI/ASA S12.60, Long, Beranek via
  Sabine at midpoint dimensions. Never fitted to the real medians: that would leak the
  test set.
- v2 materials: frequency-dependent absorption over octave bands with a per-type
  spectral tilt (soft / hard / neutral), plus per-surface variation on shoeboxes.
  10 classes x 500 rooms.
- Leave-one-room-out dropped on the synthetic set: every `room_id` is generated once, so
  rooms and samples are 1:1 and it degenerates into leave-one-out. In-sim is a stratified
  5-fold; the eval that matters is sim-to-real.
- Two-level scoring. The label set mixes functional labels (office, meeting_room) with
  acoustic ones (staircase, cathedral). A small office and a small meeting room are the
  same acoustic object. `COARSE_MAP` is defined a priori and applied to predictions after
  the fact, never before training, so the fine-coarse gap measures intra-archetype error.
- Neural input is a 64-band log-mel. A Schroeder decay curve was rejected as circular: it
  is the hand-crafted physics itself.

### Real test set

BUT 9 rooms + Aachen AIR binaural 4 + ACE Single 6 = 19 rooms; 17 rooms / 67 RIRs in the
four classes shared with the simulator (office, meeting_room, lecture_room, staircase).

Rejected on purpose: AIR phone (telephone band, 300-3400 Hz, inflates C80), ACE
multichannel and consumer captures (same capture-bias concern), MIT survey (recorded at
conversational distance for a perceptual study; median usable decay ~0.7 s, not enough
dynamic range for a T30-style fit).

### Bugs found

- `pra.measure_rt60` already extrapolates the fitted slope to -60 dB. The pipeline
  multiplied by 2 on top. Validated against ACE published ground truth: room 502 gives
  0.394 / 0.358 s against GT 0.364 / 0.310. Accuracies unaffected (constant factor on one
  feature), but every absolute RT60 reported before the fix was double.
- `utils.py` never resampled, so the 48 kHz corpora (AIR, ACE) carried energy above 8 kHz
  that no 16 kHz training file could contain.
- A `ROOM_LABELS` key matched no folder and silently dropped 155 RIRs.
- C80 divided by an empty late window on IRs shorter than 80 ms and returned +120 dB.

### Dataset ceiling

Swept the Graphi07 catalogue (20 corpora) plus MP-RIR. 11 are single-room, 3 are already
in use, MIT is out on dynamic range, REVERB and GTU-RIR have rooms but no functional
labels, the rest are off-taxonomy or extreme-only. Roughly 19 rooms is the practical
ceiling of public, functionally-labelled, full-dynamic-range RIR data. Structural
property of the field, not a gap in the search.

### Pre-registered architecture sweep (31 Jul '26)

Every neural claim rested on one 94k CNN. Registered before running, both outcomes to be
reported: same conv stack at 24k / 94k / 370k parameters plus an ImageNet-pretrained
ResNet18 (11.2M, lr 1e-4, first conv summed to 1 channel). Everything else identical.

## Five seeds: the ladder does not exist (4 Sep '26)

Re-ran after the pipeline fixes above, four nets, 5 seeds each, 120 epochs (60 for
ResNet, flat by 40), with BatchNorm statistics optionally re-estimated on the unlabelled
real set.

Coarse sim-to-real, 17 rooms / 67 RIRs, CI bootstrapped over rooms:

    XGBoost        0.806   0.627-0.964   15/17
    RandomForest   0.746   0.548-0.922   15/17
    SVM            0.746   0.557-0.914   15/17
    kNN            0.731   0.562-0.881   15/17
    ResNet18-pt    0.630 +/- 0.061         10.8
    CNN-94k        0.597 +/- 0.037         10.2
    CNN-24k        0.594 +/- 0.050         10.2
    CNN-370k       0.588 +/- 0.025         10.2
    majority       0.567                     10

- Capacity ladder gone: flat over a 460x parameter range plus pretraining. Earlier
  entries read one rung of seed noise as a trend. The drop is flat too (0.540, 0.524,
  0.527, 0.515): non-monotone, and spread smaller than the seed sd of its own terms.
- AdaBN changes nothing, so stale simulator normalization is not why the nets fail.
- Fine level: nothing clears the 0.358 baseline. Best is XGBoost 0.418 [0.206-0.644].
  Office vs meeting room stays inseparable.
- The six features are one. Pairwise |r| on the real set spans 0.53-1.00, and 0.79-1.00
  among the five excluding DRR, EDT and Ts at r = 1.00: they are integrals of the same
  decay curve. RT60 alone equals all six on coarse accuracy (0.757), and reaches 12-14
  of 17 rooms against 15 for the six together. DRR is the only weakly redundant one,
  and is confounded by acquisition: median -7.9 dB on BUT against 0.1 and 0.2 on ACE
  and AIR, tracking source-microphone distance rather than the room.

Survives: trained only on simulation, the six parameters assign the acoustic archetype of
15 of 17 unseen real rooms, and 20 neural runs on the same signal and protocol never reach
the weakest of the four feature-based classifiers. Every room-clustered interval on one
side overlaps every interval on the other, so the separation rests on that consistency,
not on any single test.

Not controlled: simulated files are written at ~1.9x RT60 against ~1.3x for real
recordings, and the fixed-length neural input exposes that transition point.

Archived here.
