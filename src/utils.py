"""Room-acoustic parameters from a room impulse response.

Six broadband ISO 3382 descriptors per RIR: RT60, EDT, C80, D50, Ts, DRR.
Every file is brought to TARGET_FS first, so the 16 kHz and 48 kHz corpora are
described over the same band.
"""

from math import gcd

import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from scipy.signal import resample_poly

TARGET_FS = 16000  # same rate as the neural pipeline, nn/ingest.py


def extract_features(wav_path: str) -> dict:
    ir, fs = sf.read(wav_path)
    if ir.ndim > 1:
        ir = ir[:, 0]  # mono

    # The real set mixes 16 kHz (BUT) and 48 kHz (AIR, ACE) while the sim set is
    # all 16 kHz. All six features are broadband, so without this the 48 kHz
    # files contribute energy above 8 kHz that no training file can contain.
    # Also keeps this pipeline on the same signal as nn/ingest.py.
    if fs != TARGET_FS:
        g = gcd(fs, TARGET_FS)
        ir = resample_poly(ir, TARGET_FS // g, fs // g)
        fs = TARGET_FS
    return features_from_ir(ir, fs)


def features_from_ir(ir, fs: int = TARGET_FS) -> dict:
    """The six parameters from an already-loaded, already-resampled IR. Split out
    so a control experiment can feed in a signal the neural pipeline shaped."""
    # RT60 broadband. pra.measure_rt60 already extrapolates the T30 slope to
    # a full -60dB decay internally (extrapolate_value_db is hardcoded there),
    # so decay_db=30 controls the fit window, not the output scale. No extra
    # *2 needed: verified against ACE Challenge ground-truth T60 (office room
    # 502, Single/Crucif-chan1: our estimate 0.394s/0.358s vs GT 0.364s/0.310s).
    rt60 = pra.measure_rt60(ir, fs=fs, decay_db=30)

    # Align to direct-sound onset: energy-based params (C80, D50, Ts, EDT, DRR)
    # must be computed from the direct arrival, not from the start of the file,
    # otherwise any pre-delay silence corrupts them.
    peak = int(np.argmax(np.abs(ir)))
    pre = int(0.0005 * fs)  # 0.5 ms before the peak
    onset = max(0, peak - pre)
    h2 = ir[onset:] ** 2
    total_energy = np.sum(h2)

    ms = int(fs / 1000)  # samples per ms

    # C80 needs energy on both sides of the 80 ms split. Some outdoor_patio IRs
    # are shorter than that, leaving an empty denominator; NaN rather than the
    # +120 dB an epsilon guard would produce.
    late = np.sum(h2[80 * ms:])
    c80 = 10 * np.log10(np.sum(h2[:80 * ms]) / late) if late > 0 else np.nan
    d50 = np.sum(h2[:50 * ms]) / (total_energy + 1e-12)
    ts = np.sum(h2 * np.arange(len(h2))) / (total_energy * fs + 1e-12)

    # EDT: early decay time from the Schroeder curve (0 to -10 dB, times 6)
    sch = np.cumsum(h2[::-1])[::-1]
    sch_db = 10 * np.log10(sch / (sch[0] + 1e-12) + 1e-12)
    below = np.nonzero(sch_db <= -10.0)[0]
    edt = 6.0 * below[0] / fs if len(below) else np.nan

    # DRR: direct (0.5 ms before to 2.5 ms after the peak) vs the rest
    direct_end = pre + int(0.0025 * fs)
    e_direct = np.sum(h2[:direct_end])
    e_late = np.sum(h2[direct_end:])
    drr = 10 * np.log10(e_direct / (e_late + 1e-12))

    return {
        "rt60": rt60,
        "edt": edt,
        "c80": c80,
        "d50": d50,
        "ts": ts,
        "drr": drr,
    }


ROOM_LABELS = {
    "Hotel_SkalskyDvur_ConferenceRoom2": "conference_room",
    "VUT_FIT_C236":                      "meeting_room",
    "VUT_FIT_L207":                      "office",
    "VUT_FIT_L212":                      "office",
    "VUT_FIT_Q301":                      "office",
    "VUT_FIT_L227":                      "staircase",
    "Hotel_SkalskyDvur_Room112":         "hotel_room",
    "VUT_FIT_E112":                      "lecture_room",
    "VUT_FIT_D105":                      "lecture_room",
}
