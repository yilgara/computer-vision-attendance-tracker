# Computer Vision Attendance Tracker

A Python-based, Streamlit-powered employee attendance system that leverages computer vision and face recognition to automate attendance tracking. This project simplifies workforce management by providing easy, accurate, and real-time attendance monitoring.

---

## Features

- **Streamlit Web App Interface**: Clean, user-friendly navigation with sidebar for quick access to all features.
- **Live Face Recognition**: Uses DeepFace and OpenCV for real-time employee identification via webcam.
- **Employee Management**: Add, update, and manage employee profiles and face data.
- **Dashboard Overview**: Visualize attendance statistics and recent activity at a glance.
- **Attendance Logging**: Automatically tracks entry/exit events, logs them with timestamps, and calculates confidence scores.
- **Attendance Reports**: View and export historical attendance records in CSV format.
- **Role Management & Settings**: Adjust system settings and manage application parameters.
- **Robust Data Handling**: Stores face embeddings, attendance logs, and employee data efficiently.
- **Session Security**: Minimizes duplicate entries with smart entry/exit buffering.


---

## How to Use

### 1. Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yilgara/computer-vision-employee-attendance.git
cd computer-vision-employee-attendance
pip install -r requirements.txt
```

### 2. Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

### 3. Main Application Pages

- **Dashboard**: Get a summary of recent attendance and overall statistics.
- **Manage Employees**: Register new employees by uploading their pictures and providing their name.
- **Live Recognition**: Activate the camera and automatically recognize faces for attendance marking.
- **Attendance Reports**: Access and export daily, weekly, or custom attendance logs.
- **Settings**: Configure the application, including recognition thresholds and entry/exit buffer.

### 4. How Attendance Works

- The system uses your webcam to detect and recognize employee faces.
- On successful recognition, it logs the entry or exit, avoiding duplicates within a short buffer period.
- All attendance events, along with timestamps and recognition confidence, are saved to a CSV file for reporting.

---

## Dependencies

- Python 3.10
- Streamlit
- OpenCV
- DeepFace
- pandas, numpy, pickle, csv, etc.

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

---

**Deployed Application:**  
You can access the live system at: [https://github.com/yilgara/computer-vision-attendance-tracker](https://github.com/yilgara/computer-vision-attendance-tracker)
