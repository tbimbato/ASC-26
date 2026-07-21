"""RIR ingestion for the neural baseline.

Everything the network sees goes through load_ir() so the whole pipeline is
homogeneous. Checked on 2026-07-21: sim wavs are all 16 kHz mono, real wavs
are mixed 16/48 kHz (BUT is 16k, AIR/ACE are 48k), all mono. Durations vary
wildly (sim up to ~10 s for cathedral/gas_tank, real up to ~3.1 s), so we cut
or zero-pad to a fixed length.

Homogenization steps, in order:
  1. mono (defensive, files are already mono)
  2. resample to TARGET_FS
  3. align to the direct-sound onset (same trick as utils.py: peak minus
     0.5 ms). Pre-delay silence is an artifact of the recording chain, not
     of the room, the network should not see it.
  4. peak-normalize. Absolute gain is arbitrary (mic, distance, file format),
     not room information. Relative decay is preserved, which is what matters.
  5. zero-pad or truncate to FIXED_LEN_S seconds. Truncation eats the tail of
     the most reverberant sim rooms; documented limitation, revisit if the
     cathedral/tank classes suffer in-sim.
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_FS = 16000
FIXED_LEN_S = 2.0   # seconds; real max is 3.1 s, sim overlap classes are shorter
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
