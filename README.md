<img width="1919" height="969" alt="image" src="https://github.com/user-attachments/assets/5a126a2a-ca94-40ed-9e4b-3f23fbb1aba1" />🫀 ECG Arrhythmia Detection System

AI-Assisted Cardiac Risk Analysis Web Application

📌 Overview

The ECG Arrhythmia Detection System is a web-based application that analyzes ECG (Electrocardiogram) signals using Deep Learning (CNN + LSTM) to detect abnormal heart rhythms.

It helps in:

Identifying Normal vs Abnormal ECG
Calculating Heart Rate (BPM)
Determining Risk Level
Generating a Medical PDF Report
Sending the report to a Cardiologist via Email
🎯 Problem Statement
ECG interpretation requires expertise
Manual analysis is time-consuming
Early detection of heart issues is critical

👉 This system uses AI to assist in quick preliminary diagnosis

🧠 How It Works (Simple Explanation)
User uploads ECG CSV file
Signal is filtered & cleaned
Peaks (heartbeats) are detected
Model predicts each heartbeat
Average probability is calculated
System outputs:
Diagnosis
Risk level
BPM (Heart Rate)
🛠️ Tech Stack
🔹 Backend
Python
Flask
🔹 Machine Learning
TensorFlow / Keras
CNN + LSTM Model
SciPy (Signal Processing)
🔹 Frontend
HTML
CSS
Bootstrap
🔹 Additional Tools
Matplotlib → ECG Graph
ReportLab → PDF Generation
SMTP → Email Sending



