"""RIR ingestion for the neural baseline.

Everything the network sees goes through load_ir(), so the whole pipeline is
homogeneous. Sim wavs are 16 kHz mono; the real set mixes 16 kHz (BUT) and
48 kHz (AIR, ACE). Durations range from ~0.1 s to ~10 s, hence the fixed length.

Steps, in order:
  1. mono
  2. resample to TARGET_FS
  3. align to the direct-sound onset (peak minus 0.5 ms, as in utils.py), so
     pre-delay silence does not shift everything
  4. peak-normalize: absolute gain carries no room information, the relative
     decay does
  5. zero-pad or truncate to FIXED_LEN_S. Padding leaves an edge whose position
     tracks the original duration, which in the sim set correlates with RT60;
     see the Limitations section of the README for what that costs.
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_FS = 16000
FIXED_LEN_S = 3.2   # seconds; real max is 3.1 s, sim overlap classes are shorter
FIXED_LEN = int(TARGET_FS * FIXED_LEN_S) 


def load_ir(wav_path: str) -> np.ndarray:
    """Wav file -> homogenized 1-D float array of length FIXED_LEN."""
    ir, fs = sf.read(wav_path)
    if ir.ndim > 1:
        ir = ir[:, 0]

    if fs != TARGET_FS:
        # rational resampling, e.g. 48k -> 16k is just 1/3
        from math import gcd
        g = gcd(fs, TARGET_FS)
        ir = resample_poly(ir, TARGET_FS // g, fs // g)

    # align to direct sound
    peak = int(np.argmax(np.abs(ir)))
    onset = max(0, peak - int(0.0005 * TARGET_FS))
    ir = ir[onset:]

    # gain is not information
    m = np.max(np.abs(ir))
    if m > 0:
        ir = ir / m

    # fixed length
    if len(ir) >= FIXED_LEN:
        ir = ir[:FIXED_LEN]
    else:
        ir = np.pad(ir, (0, FIXED_LEN - len(ir)))
    return ir.astype(np.float32)
