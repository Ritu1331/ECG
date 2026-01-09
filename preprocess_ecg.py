import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Path to ECG data
DATA_PATH = "data/ecg_arrhythmia_project/mit-bih-arrhythmia-database-1.0.0"

# Load ECG
record = wfdb.rdrecord(f"{DATA_PATH}/100")
signal = record.p_signal[:, 0]

# Bandpass filter function
def bandpass_filter(sig, fs=360, low=0.5, high=40):
    nyq = 0.5 * fs
    b, a = butter(1, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

filtered_signal = bandpass_filter(signal)

# Plot raw vs filtered ECG
plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(signal[:2000])
plt.title("Raw ECG Signal")

plt.subplot(2,1,2)
plt.plot(filtered_signal[:2000])
plt.title("Filtered ECG Signal")

plt.tight_layout()
plt.show()
