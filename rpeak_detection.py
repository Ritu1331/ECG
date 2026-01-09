import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Load ECG
DATA_PATH = "data/ecg_arrhythmia_project/mit-bih-arrhythmia-database-1.0.0"
record = wfdb.rdrecord(f"{DATA_PATH}/100")
signal = record.p_signal[:, 0]

# Bandpass filter (same as before)
def bandpass_filter(sig, fs=360, low=0.5, high=40):
    nyq = 0.5 * fs
    b, a = butter(1, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

filtered_signal = bandpass_filter(signal)

# 🔥 R-peak detection
peaks, _ = find_peaks(
    filtered_signal,
    distance=150,      # minimum distance between heartbeats
    height=0.5         # detect only strong peaks
)

print("Total heartbeats detected:", len(peaks))

# Plot ECG with R-peaks
plt.figure(figsize=(12,4))
plt.plot(filtered_signal[:2000], label="Filtered ECG")
plt.plot(peaks[peaks < 2000], 
         filtered_signal[peaks[peaks < 2000]], 
         "ro", label="R-peaks")
plt.legend()
plt.title("R-Peak Detection")
plt.show()
