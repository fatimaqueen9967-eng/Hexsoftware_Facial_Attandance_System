import gradio as gr
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime
import pandas as pd

# ---------------- LOAD DATA ----------------
path = "images"
images = []
classNames = []

for img_name in os.listdir(path):
    img = face_recognition.load_image_file(f"{path}/{img_name}")
    images.append(img)
    classNames.append(os.path.splitext(img_name)[0])


def encode_faces(images):
    encodings = []
    for img in images:
        enc = face_recognition.face_encodings(img)[0]
        encodings.append(enc)
    return encodings


known_encodings = encode_faces(images)

# ---------------- ATTENDANCE ----------------
def mark_attendance(name):
    file = "attendance.csv"

    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    with open(file, "r") as f:
        data = f.readlines()
        for line in data:
            if line.split(",")[0] == name:
                return "Already Marked"

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time])

    return "Marked"


# ---------------- FACE RECOGNITION ----------------
def recognize_face(image):

    img = np.array(image)

    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    results = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = classNames[best_match]
            status = mark_attendance(name)
        else:
            status = "Not Recognized"

        results.append([name, status])

    if len(results) == 0:
        return [["No Face Detected", ""]]

    return results


# ---------------- DOWNLOAD ATTENDANCE ----------------
def download_csv():
    return "attendance.csv"


# ---------------- UI ----------------
with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🧑‍💼 Face Recognition Attendance System")

    with gr.Row():

        with gr.Column():
            input_img = gr.Image(
                type="pil",
                height=400,
                width=400
            )

            btn = gr.Button("Detect & Mark Attendance", variant="primary")

            download_btn = gr.Button("⬇ Download Attendance CSV")

        with gr.Column():
            output = gr.Dataframe(
                headers=["Name", "Status"],
                datatype=["str", "str"],
                interactive=False
            )

            file_output = gr.File(label="Download File")

    btn.click(fn=recognize_face, inputs=input_img, outputs=output)
    download_btn.click(fn=download_csv, inputs=None, outputs=file_output)

app.launch()