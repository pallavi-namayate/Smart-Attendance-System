# Smart Face Recognition Attendance System

A robust, full-stack Flask web application that automates student and teacher attendance tracking using modern Computer Vision and Face Recognition techniques. The system logs daily records into CSV files and features automatic absentee SMS notifications for parents via Twilio Integration.

---

## 🚀 Features

### 1. Dual Attendance Recording Modes
* **Real-time Live Camera View:** Launch an interactive webcam interface right from your browser to punch live attendance instantly.
* **Group / Class Photo Upload:** Upload a single class or team picture. The application automatically handles multi-face detection, extracts individual faces, processes them at multiple scales to prevent missing profiles, and accurately labels everybody in the picture.

### 2. Intelligent Computer Vision Backend
* **Multi-Scale & Greedy NMS Filtering:** Detections are evaluated at multiple resolution scales and filtered through an optimized Non-Maximum Suppression (NMS) layer to eliminate duplicate bounding boxes around dense or challenging facial angles.
* **Dynamic Matching Thresholds:** Uses both strict and relaxed fallback threshold variables to match recognized faces reliably against existing digital records.
* **Automated Verification Audit Reports:** For every group photo uploaded, the system saves an matching JSON report and creates a bounding-box annotated version of the image for visual verification.

### 3. Comprehensive Database Logs & Administration
* **Teacher & Student Directories:** Simple forms to register individual records into dedicated CSV datasets (`students.csv` and `teachers.csv`).
* **Daily Records Dashboard:** Interactive admin panel showing real-time attendance logs filtered by date, directly pointing out missing absentees alongside parent contacts.

### 4. Direct SMS Integration
* **Automated Notifications:** When a student is flagged absent after an evaluation pass, the backend leverages Twilio to dispatch immediate SMS alerts directly to the linked parent phone numbers.

---

## 🛠️ Technology Stack

* **Backend Framework:** Python 3, Flask, Jinja2 template engine
* **Computer Vision & ML:** OpenCV (`opencv-python`), `dlib`, and `face-recognition`
* **Data Processing:** Pandas, NumPy
* **Communication Services:** Twilio REST Client API

---

## 📁 Directory Structure Breakdown

```text
├── class_photos/              # Saved group uploads, generated audit reports, and annotations
├── Training images/           # Registered profile photographs categorized by name (.png/.jpg)
├── templates/                 # UI HTML interfaces (Dashboard, Forms, Main panel, etc.)
├── static/                    # Layout styling sheets (CSS, JS structures, vendor packages)
├── app.py                     # Central Flask orchestration and core algorithmic pipelines
├── attendance.csv             # Automated rolling database tracking daily student attendance
├── attendance_teachers.csv    # Automated rolling database tracking daily teacher attendance
├── students.csv               # Student metadata dataset containing relative guardian contacts
├── teachers.csv               # Teacher tracking metadata and related academic subjects
├── requirements.txt           # Explicit Python virtual environment dependencies
└── README.md                  # Detailed platform documentation
