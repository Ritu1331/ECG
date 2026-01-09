import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# Path to dataset
DATA_PATH = "data/ecg_arrhythmia_project/mit-bih-arrhythmia-database-1.0.0"

# Load ECG signal and annotations
record = wfdb.rdrecord(f"{DATA_PATH}/100")
annotation = wfdb.rdann(f"{DATA_PATH}/100", "atr")

signal = record.p_signal[:, 0]
symbols = annotation.symbol  # Doctor labels

# Bandpass filter
def bandpass_filter(sig, fs=360, low=0.5, high=40):
    nyq = 0.5 * fs
    b, a = butter(1, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

filtered_signal = bandpass_filter(signal)

# R-peak detection
peaks, _ = find_peaks(filtered_signal, distance=150, height=0.5)

# Heartbeat segmentation + labeling
beats = []
labels = []
window = 90

for peak, symbol in zip(peaks, symbols):
    if peak - window >= 0 and peak + window < len(filtered_signal):
        beat = filtered_signal[peak-window : peak+window]
        beats.append(beat)

        # 🔥 Labeling rule
        if symbol == 'N':
            labels.append(0)  # Normal
        else:
            labels.append(1)  # Abnormal

beats = np.array(beats)
labels = np.array(labels)

print("Total heartbeats:", beats.shape[0])
print("Normal beats:", np.sum(labels == 0))
print("Abnormal beats:", np.sum(labels == 1))
