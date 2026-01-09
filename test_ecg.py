import wfdb
import matplotlib.pyplot as plt

# Correct path to MIT-BIH dataset folder
DATA_PATH = "data/ecg_arrhythmia_project/mit-bih-arrhythmia-database-1.0.0"

# Load ECG record 100
record = wfdb.rdrecord(f"{DATA_PATH}/100")
annotation = wfdb.rdann(f"{DATA_PATH}/100", "atr")

# ECG signal (lead 0)
signal = record.p_signal[:, 0]

print("Signal length:", len(signal))
print("First 10 beat labels:", annotation.symbol[:10])

# Plot ECG signal
plt.figure(figsize=(12, 4))
plt.plot(signal[:2000])
plt.title("Raw ECG Signal (Record 100)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
