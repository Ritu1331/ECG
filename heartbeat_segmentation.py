import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Load ECG
DATA_PATH = "data/ecg_arrhythmia_project/mit-bih-arrhythmia-database-1.0.0"
record = wfdb.rdrecord(f"{DATA_PATH}/100")
signal = record.p_signal[:, 0]

# Bandpass filter
def bandpass_filter(sig, fs=360, low=0.5, high=40):
    nyq = 0.5 * fs
    b, a = butter(1, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

filtered_signal = bandpass_filter(signal)

# R-peak detection
peaks, _ = find_peaks(filtered_signal, distance=150, height=0.5)

# 🔥 Heartbeat segmentation
beats = []
window = 90   # 90 samples before and after R-peak

for peak in peaks:
    if peak - window >= 0 and peak + window < len(filtered_signal):
        beat = filtered_signal[peak - window : peak + window]
        beats.append(beat)

beats = np.array(beats)

print("Total heartbeats extracted:", beats.shape[0])

# Plot one heartbeat
plt.plot(beats[0])
plt.title("Single Heartbeat Segment")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
