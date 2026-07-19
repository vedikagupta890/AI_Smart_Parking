# AI Smart Parking System

An AI-powered Smart Parking System that detects vehicles in a parking lot using YOLO and OpenCV, calculates parking occupancy in real time, and displays live statistics through a Flask-based web dashboard.



https://github.com/user-attachments/assets/862b5267-e2bf-409d-8eca-3556940deda4



---

## Overview

The AI Smart Parking System is a computer vision application that automates parking occupancy monitoring using object detection.

The system processes a video feed, detects vehicles using a YOLO model, calculates occupied and available parking spaces, and streams the processed video along with live parking statistics to a responsive web interface.

This project demonstrates the integration of Artificial Intelligence, Computer Vision, and Full-Stack Web Development.

---

## Features

- Real-time vehicle detection using YOLO
- Live parking occupancy monitoring
- Flask backend with REST APIs
- MJPEG video streaming
- Responsive web dashboard
- Live parking statistics
- Modular and scalable architecture
- Easy deployment and configuration

---

## Tech Stack

### Backend

- Python
- Flask
- OpenCV
- NumPy

### AI / Computer Vision

- YOLO
- ONNX Runtime (or Ultralytics YOLO if applicable)

### Frontend

- HTML5
- CSS3
- JavaScript

---

## System Architecture

```
             Video Source
                   │
                   ▼
          SmartParkingPipeline
                   │
         Vehicle Detection (YOLO)
                   │
          Occupancy Calculation
                   │
             Flask Backend
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
 Video Stream API         Status API
      │                         │
      └────────────┬────────────┘
                   ▼
             Web Dashboard
```

---

## Project Structure (Main Files)

```text
AI-Smart-Parking/
│
├── app.py
├── parking.py
├── slot_detector_yolo.py
├── requirements.txt
│
├── models/
│   └── best.onnx
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│

```

---

## Workflow

1. Read video frames.
2. Perform vehicle detection using YOLO.
3. Identify occupied parking spaces.
4. Calculate parking statistics.
5. Stream annotated video.
6. Update the dashboard in real time.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/video_feed` | GET | Live video stream |
| `/api/status` | GET | Parking statistics |
| `/health` | GET | Health check |

---

## Installation

### Clone the repository

```bash
git clone https://github.com/<username>/<repository>.git

cd <repository>
```

### Create a virtual environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask server.

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## Screenshots

### Dashboard

<img width="959" height="470" alt="image" src="https://github.com/user-attachments/assets/ce12c3fe-f6ea-43dc-a2db-9e06f347089c" />

---

### Live Vehicle Detection

<img width="959" height="461" alt="image" src="https://github.com/user-attachments/assets/7fac088e-fa86-4725-a120-e97e46d740ea" />

---

### Parking Statistics

<img width="916" height="418" alt="image" src="https://github.com/user-attachments/assets/c1557a9b-1f54-441e-b0b3-10e2bf50fb1a" />

---

## Future Improvements

- Multi-camera support
- Database integration
- License Plate Recognition (LPR)
- Cloud deployment
- Mobile application
- Historical parking analytics
- User authentication
- GPU acceleration

---

## Learning Outcomes

This project helped in gaining practical experience with:

- Computer Vision
- Object Detection
- Flask Web Development
- REST API Design
- OpenCV
- Real-time Video Processing
- Software Architecture
- Full-Stack Integration

---

## License

This project is licensed under the MIT License.

---

## Author

**Vedika Gupta**

GitHub: https://github.com/vedikagupta890

LinkedIn: https://www.linkedin.com/in/vedika-gupta-088655324/
