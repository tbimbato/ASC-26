import numpy as np
import soundfile as sf
import pyroomacoustics as pra

def extract_features(wav_path: str) -> dict:
    ir, fs = sf.read(wav_path)
    if ir.ndim > 1:
        ir = ir[:, 0]  # mono

    # RT60 broadband
    rt60 = pra.measure_rt60(ir, fs=fs, decay_db=30) * 2  # T30 → T60

    # Energy decay curve per calcolare EDT, C80, D50, Ts
    h2 = ir ** 2
    cumsum = np.cumsum(h2)
    total_energy = cumsum[-1]

    ms = int(fs / 1000)  # sample per ms

    c80 = 10 * np.log10(np.sum(h2[:80*ms]) / (np.sum(h2[80*ms:]) + 1e-10))
    d50 = np.sum(h2[:50*ms]) / (total_energy + 1e-10)
    ts  = np.sum(h2 * np.arange(len(h2))) / (total_energy * fs + 1e-10)

    return {
        "rt60": rt60,
        "c80": c80,
        "d50": d50,
        "ts": ts,
    }


ROOM_LABELS = {
    "Hotel_SkalskyDvur_ConferenceRoom2": "conference_room",
    "VUT_FIT_C236":                      "meeting_room",
    "VUT_FIT_L207":                      "office",
    "VUT_FIT_L212":                      "office",
    "VUT_FIT_Q301":                      "office",
    "VUT_FIT_L227":                      "staircase",
    "VUT_FIT_R112":                      "hotel_room",
    "VUT_FIT_E112":                      "lecture_room",
    "VUT_FIT_D105":                      "lecture_room",
}
