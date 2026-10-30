![CI](https://github.com/azzahzayn/recall-an-ar-based-facial-recognition-system-for-elderly/actions/workflows/ci.yml/badge.svg)  

# Real-Time Facial Recognition System for People with Prosopagnosia

## 📌 About the Project

This project is a real-time facial recognition system designed to help people with **prosopagnosia (face blindness)** recognize and identify familiar people.

The system uses a webcam to detect faces, extract facial features, and compare them with previously stored information. It also uses depth estimation to add another layer of information for facial matching.

The goal is to help users identify familiar people more easily.

## ✨ Features

- **Real-Time Face Detection:** Detects faces from the webcam feed.
- **Facial Recognition:** Uses facial embeddings to recognize registered people.
- **Depth Estimation:** Uses Depth Anything V2 to estimate depth information from camera frames.
- **Multimodal Matching:** Combines facial and depth-related features for recognition.
- **Face Enrollment:** Stores information about registered individuals for future recognition.
- **Database Storage:** Uses SQLite to store identities and facial and depth embeddings.
- **Live Camera Feed:** Displays the camera feed and recognition results.
- **Recognition Feedback:** Displays recognized identities on screen.
- **Performance Optimization:** Runs depth estimation in a background thread to reduce delays in the video processing loop.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe Face Mesh | Facial landmark detection |
| ArcFace | Facial feature extraction and recognition |
| Depth Anything V2 | Monocular depth estimation |
| ONNX Runtime | Model inference |
| SQLite | Local database storage |
| PyGrabber | DirectShow camera device enumeration |

## ⚙️ Requirements

- Python installed on your computer
- A Windows computer with a working USB camera
- Waveshare OV5640 5MP USB camera
- Required Python packages
- Model weights and configuration files required by the project

The project is configured to require the external camera rather than automatically falling back to the built-in laptop webcam.
