# ASC-26: Acoustic Space Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21771185.svg)](https://doi.org/10.5281/zenodo.21771185)

Room-type classification from room impulse responses, trained exclusively on simulated
rooms and evaluated on real recordings from rooms never seen during training.

A room-acoustics simulator (`pyroomacoustics`) generates 5,000 labelled RIRs over
10 room types with frequency-dependent per-surface absorption over four geometry
families. Classifiers are fitted on this synthetic set and tested on 19 real rooms
assembled from BUT ReverbDB, Aachen AIR and the ACE Challenge. Evaluation is reported
at two levels: fine (functional labels as recorded) and coarse (acoustic archetypes,
collapsed after prediction, defined a priori). The four classes present in both domains
are office, meeting room, lecture room and stairwell; the real subset restricted to
these is 17 rooms, 67 RIRs.

Two model families are compared on the identical protocol: six ISO 3382 parameters
(RT60, EDT, C80, D50, Ts, DRR) with four standard classifiers, and log-mel
spectrograms with four convolutional architectures spanning 24k to 11M parameters,
each over five seeds.

## Results

Coarse sim-to-real. Confidence intervals are 95%, bootstrapped over rooms rather than
RIRs; room-level accuracy is one majority vote per room.

| Model | Accuracy | 95% CI | Rooms |
|---|---|---|---|
| XGBoost (ISO 3382) | 0.806 | 0.627-0.964 | 15/17 |
| RandomForest | 0.746 | 0.548-0.922 | 15/17 |
| SVM | 0.746 | 0.557-0.914 | 15/17 |
| kNN | 0.731 | 0.562-0.881 | 15/17 |
| ResNet18 (ImageNet-pretrained) | 0.630 ± 0.061 | | 10.8/17 |
| CNN-94k | 0.597 ± 0.037 | | 10.2/17 |
| CNN-24k | 0.594 ± 0.050 | | 10.2/17 |
| CNN-370k | 0.588 ± 0.025 | | 10.2/17 |
| Majority class | 0.567 | | 10/17 |

Across 20 neural runs the coarse accuracy ranges from 0.537 to 0.702; none reaches the
weakest of the four feature-based classifiers.

At the fine level no model separates from the 0.358 majority baseline. The best
headline model is XGBoost at 0.418 (CI 0.206-0.644); the highest fine accuracy
anywhere in the study is 0.463, from RandomForest on RT60 alone. Office and
meeting room are not distinguishable from the acoustics alone.

Three controls:

Capacity: accuracy is flat from 24k to 11M parameters and under ImageNet pretraining,
within the seed spread.

Domain normalization: recomputing BatchNorm statistics on the unlabelled real set
(AdaBN) lowers coarse accuracy for all four architectures.

Feature redundancy: the six parameters are integrals of the same energy decay
  curve. On the real set their pairwise |r| spans 0.53-1.00, and 0.79-1.00 among
  the five excluding DRR, with EDT and Ts at r = 1.00. RT60 alone matches the full
  set on coarse accuracy (0.757 vs 0.757), though at room level the six together
  reach 15/17 against 12-14/17 for RT60 alone. DRR is the least
  redundant parameter (r = 0.53 with RT60) and is confounded by acquisition: its median
  is -7.9 dB on BUT against 0.1 and 0.2 dB on ACE and AIR, tracking source-microphone
  distance rather than the room.

## Limitations

The real evaluation covers 17 rooms. Room-clustered intervals are correspondingly wide
and no single interval separates the two model families; the separation rests on
consistency across 4 feature-based models and 20 neural runs, and on the room-level
figures.

Simulated files are written at approximately 1.9 x RT60 in length against 1.3 x RT60
for the real recordings. The neural arm receives fixed-length inputs and can read the
signal-to-floor transition point; this cue is not controlled for and does not affect the
feature-based arm.

The training-time noise-floor augmentation is not applied at evaluation, so the in-sim
test split is scored under a mild distribution shift.

## Data

The synthetic set (5,000 labelled RIRs, 10 room types, 16 kHz PCM-16) is archived with
a datasheet at [10.5281/zenodo.21771185](https://doi.org/10.5281/zenodo.21771185) and is
usable independently of the analysis in this repository. The real evaluation set is
assembled from third-party corpora. Their audio is not redistributed here;
`src/real_test.py` rebuilds the manifest from local copies, and `data/real/` holds
only the extracted parameter table and the file listing.

The real rooms come from three corpora, each obtainable from its own publisher under
its own terms:

- BUT ReverbDB. Szoke, Skacel, Mosner, Paliesek, Cernocky, "Building and evaluation
  of a real room impulse response dataset", IEEE JSTSP 13(4), 2019.
- Aachen AIR. Jeub, Schafer, Vary, "A binaural room impulse response database for the
  evaluation of dereverberation algorithms", DSP 2009.
- ACE Challenge. Eaton, Gaubitch, Moore, Naylor, "Estimation of room acoustic
  parameters: the ACE challenge", IEEE/ACM TASLP 24(10), 2016.

Simulation uses pyroomacoustics: Scheibler, Bezzam, Dokmanic, "Pyroomacoustics: a
Python package for audio room simulation and array processing algorithms",
ICASSP 2018.

The MIT license covers the code and the synthetic data. The tables under `data/real/`
and `results/` are derived from the three corpora above and remain subject to their
terms.

## Pipeline

| Step | Script |
|------|--------|
| Room-type taxonomy and geometry | `src/room_types.py` |
| Synthetic RIR generation | `src/simulate.py` |
| Feature extraction, ISO 3382 | `src/utils.py` |
| Features from a manifest | `src/build_dataset.py` |
| Real test set assembly | `src/real_test.py` |
| Feature-based benchmark | `src/train.py` |
| Neural benchmark | `src/nn/train_nn.py` |

Metrics land in `results/metrics.csv` and `results/metrics_nn.csv`, per-sample
predictions on the real set in `results/preds_nn.csv`. Setup in
[INSTRUCTIONS.md](INSTRUCTIONS.md), working notes in [notes.md](notes.md).

## Status

Archived. Earlier revisions reported a transfer gap that widened with network capacity;
that result was obtained from single-seed runs, lies within the seed spread under
five-seed evaluation, and has been removed.

MSc side project, Uppsala University (Machine Learning & Statistics).

## Citation

```bibtex
@dataset{bimbato2026asc26,
  author    = {Bimbato, Tommi},
  title     = {{ASC-26: A Synthetic Room Impulse Response Dataset
                for Room-Type Classification}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1},
  doi       = {10.5281/zenodo.21771185},
  url       = {https://doi.org/10.5281/zenodo.21771185}
}
```
