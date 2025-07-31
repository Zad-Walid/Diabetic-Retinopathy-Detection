from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
from PIL import Image

def generate_pdf(original_path, gradcam_path, diagnosis, save_path="diagnosis_report.pdf"):
    c = canvas.Canvas(save_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Eye Disease Diagnosis Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 100, f"Diagnosis: {diagnosis}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 140, "Original Image:")
    c.drawImage(original_path, 50, height - 380, width=200, preserveAspectRatio=True)

    c.drawString(300, height - 140, "Grad-CAM Heatmap:")
    c.drawImage(gradcam_path, 300, height - 380, width=200, preserveAspectRatio=True)

    c.save()
    return save_path