import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from collections import Counter

# ===============================
# CONFIG
# ===============================
DATA_PATH = "data/ecg_arrhythmia_project/mit-bih-arrhythmia-database-1.0.0"

RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119",
    "121","122","123","124",
    "200","201","202","203","205","207","208","209","210",
    "212","213","214","215","217","219","220","221","222","223",
    "228","230","231","232","233","234"
]


FS = 360          # Sampling frequency
WINDOW = 90       # 180 samples per beat (90 before + 90 after)

# ===============================
# BANDPASS FILTER
# ===============================
def bandpass_filter(sig, fs=FS, low=0.5, high=40):
    nyq = 0.5 * fs
    b, a = butter(1, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

# ===============================
# DATASET CONTAINER
# ===============================
beats = []
labels = []

# ===============================
# LOOP THROUGH MULTIPLE PATIENTS
# ===============================
for rec in RECORDS:
    print(f"Processing record {rec}...")

    record = wfdb.rdrecord(f"{DATA_PATH}/{rec}")
    annotation = wfdb.rdann(f"{DATA_PATH}/{rec}", "atr")

    signal = record.p_signal[:, 0]
    symbols = annotation.symbol
    ann_samples = annotation.sample

    # Filter ECG
    filtered_signal = bandpass_filter(signal)

    # R-peak detection
    peaks, _ = find_peaks(filtered_signal, distance=150, height=0.5)

    # Match peaks with annotations
    for peak in peaks:
        # find closest annotation to this peak
        idx = np.argmin(np.abs(ann_samples - peak))
        symbol = symbols[idx]

        if peak - WINDOW >= 0 and peak + WINDOW < len(filtered_signal):
            beat = filtered_signal[peak - WINDOW : peak + WINDOW]
            beats.append(beat)

            # Binary labeling
            labels.append(0 if symbol == "N" else 1)

# ===============================
# CONVERT TO NUMPY
# ===============================
X = np.array(beats).reshape(len(beats), 180, 1)
y = np.array(labels)

print("\nDataset summary:")
print("Total beats:", len(y))
print("Class distribution:", Counter(y))

# ===============================
# SAVE DATASET (VERY IMPORTANT)
# ===============================
np.save("X_ecg.npy", X)
np.save("y_ecg.npy", y)

print("\nSaved files:")
print("X_ecg.npy, y_ecg.npy")
