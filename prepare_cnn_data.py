import numpy as np
from sklearn.model_selection import train_test_split
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

# Load ECG
DATA_PATH = "data/ecg_arrhythmia_project/mit-bih-arrhythmia-database-1.0.0"
record = wfdb.rdrecord(f"{DATA_PATH}/100")
annotation = wfdb.rdann(f"{DATA_PATH}/100", "atr")

signal = record.p_signal[:, 0]
symbols = annotation.symbol

# Filter
def bandpass_filter(sig, fs=360, low=0.5, high=40):
    nyq = 0.5 * fs
    b, a = butter(1, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

filtered_signal = bandpass_filter(signal)

# R-peaks
peaks, _ = find_peaks(filtered_signal, distance=150, height=0.5)

# Segment + label
beats = []
labels = []
window = 90

for peak, symbol in zip(peaks, symbols):
    if peak-window >= 0 and peak+window < len(filtered_signal):
        beats.append(filtered_signal[peak-window:peak+window])
        labels.append(0 if symbol == 'N' else 1)

beats = np.array(beats)
labels = np.array(labels)

# 🔥 Prepare for CNN
X = beats.reshape(beats.shape[0], beats.shape[1], 1)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)
