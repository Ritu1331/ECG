from flask import Flask, render_template, request, send_file, redirect, flash, session
import os
from utils import predict_ecg_csv, generate_pdf, send_report_email
from datetime import datetime

app = Flask(__name__)
app.secret_key = "secret123"  # required for session & flash

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# MAIN ROUTE: ANALYZE ECG
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # -------- File upload --------
        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # -------- ECG prediction --------
        result, probability, risk, bpm, hr_status = predict_ecg_csv(path)

        # -------- Patient details --------
        patient = {
            "name": request.form["name"],
            "age": request.form["age"],
            "gender": request.form["gender"],
            "history": request.form["history"]
        }

        timestamp = datetime.now().strftime("%d %b %Y | %I:%M %p")

        # -------- Generate PDF --------
        generate_pdf(patient, result, probability, risk, bpm, hr_status)

        # -------- Store data in session (for email) --------
        session["patient_name"] = patient["name"]
        session["age"] = patient["age"]
        session["gender"] = patient["gender"]
        session["risk"] = risk
        session["bpm"] = bpm
        session["probability"] = probability

        return render_template(
            "index.html",
            result=result,
            probability=probability,
            risk=risk,
            bpm=bpm,
            hr_status=hr_status,
            timestamp=timestamp,
            pdf_ready=True
        )

    return render_template("index.html")


# =========================
# DOWNLOAD PDF
# =========================
@app.route("/download-report")
def download_report():
    return send_file("reports/ecg_report.pdf", as_attachment=True)


# =========================
# SEND EMAIL TO DOCTOR
# =========================
@app.route("/send-email", methods=["POST"])
def send_email():
    doctor_email = request.form["doctor_email"]

    try:
        send_report_email(
            doctor_email=doctor_email,
            patient_name=session["patient_name"],
            age=session["age"],
            gender=session["gender"],
            risk=session["risk"],
            bpm=session["bpm"],
            probability=session["probability"],
            pdf_path="reports/ecg_report.pdf"
        )
        flash("Report sent to doctor successfully ✅")
    except Exception as e:
        flash(f"Failed to send email ❌ {e}")

    return redirect("/")


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

