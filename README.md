![CI](https://github.com/azzahzayn/recall-an-ar-based-facial-recognition-system-for-elderly/actions/workflows/ci.yml/badge.svg)  

# Recall: AR-Based Facial Recognition System for Elderly People

## Overview

Recall is an AI-powered facial recognition system designed to assist elderly people, especially those experiencing difficulty recognizing familiar faces. The system identifies registered individuals using facial recognition and combines facial appearance with depth information to improve identity matching.

The system uses face detection, facial alignment, facial embeddings, and depth estimation to recognize registered individuals in real time.

## Features

- **Face Detection:** Detects faces from camera input using MediaPipe Face Mesh.
- **Facial Recognition:** Uses ArcFace to generate facial embeddings and identify registered individuals.
- **Depth Estimation:** Uses Depth Anything V2 to estimate depth information from facial images.
- **3D Facial Representation:** Combines RGB information with estimated depth to represent facial structure.
- **Identity Management:** Stores registered identities and their embeddings in an SQLite database.
- **Real-Time Processing:** Captures camera frames and processes detected faces for recognition.
- **Embedding Storage:** Saves facial and depth embeddings for later identity matching.

## Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Face Detection and Alignment | MediaPipe Face Mesh |
| Facial Recognition | ArcFace |
| Depth Estimation | Depth Anything V2 |
| Model Inference | ONNX Runtime |
| Image Processing | OpenCV |
| Database | SQLite |
| Testing | Pytest |
| Version Control | Git and GitHub |
| Continuous Integration | GitHub Actions |

## System Architecture

The system follows a modular processing pipeline:

1. **Frame Capture:** Captures images using a connected camera.
2. **Face Detection:** Detects facial landmarks and identifies face regions.
3. **Face Alignment:** Aligns detected faces into a consistent orientation.
4. **Depth Estimation:** Estimates depth information using Depth Anything V2.
5. **Embedding Generation:** Generates facial embeddings using ArcFace and stores the corresponding depth representations.
6. **Identity Matching:** Compares the generated embeddings with registered identities.
7. **Database Management:** Stores identity information and embeddings in SQLite.
8. **Recognition Output:** Identifies a matching registered person when the similarity score meets the configured threshold.

## Project Structure

```text
recall/
├── .github/
│   └── workflows/
│       └── ci.yml
├── common/
├── configs/
├── modules/
├── scripts/
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Database

Recall uses SQLite to store registered identities and their associated facial representations.

The database maintains information such as:

- Registered identities
- Facial embeddings
- Depth embeddings
- Model and feature version information

The stored embeddings allow newly detected faces to be compared with previously registered individuals.

The exact database location depends on the project configuration.

## Testing

The project uses Pytest for automated testing.

Run the tests from the project root:

```bash
python -m pytest -v
```

To run a specific test file:

```bash
python -m pytest -v tests/test_capture.py
```

The tests help verify individual components and identify errors during development.

## Continuous Integration

Recall uses GitHub Actions to automate testing when changes are pushed to GitHub or pull requests are created.

The workflow is configured in:

```text
.github/workflows/ci.yml
```

The CI pipeline performs the following steps:

1. Checks out the repository.
2. Sets up Python.
3. Installs project dependencies.
4. Executes the automated tests.
5. Reports whether the tests pass or fail.

This helps detect problems before changes are merged into the main branch.

## Deployment

The application can be deployed using a suitable cloud platform, such as Render or AWS, for supported backend services.

Docker can be used to package the application and its dependencies for consistent deployment. GitHub Actions can also be extended to automate deployment after successful testing.

However, real-time facial recognition requires camera input and sufficient processing resources. Camera access, model compatibility, and hardware acceleration must be considered when selecting a deployment environment.

## Applications

- Assisting elderly people in recognizing familiar individuals.
- Supporting people who experience difficulty recognizing faces.
- Maintaining a database of registered individuals.
- Exploring facial recognition using both appearance and estimated depth information.

## Future Enhancements

- Improve recognition speed and real-time performance.
- Optimize depth estimation for CPU and GPU execution.
- Improve recognition accuracy under different lighting conditions.
- Add a user-friendly interface for registering and identifying individuals.
- Support additional camera configurations.
- Improve privacy and security for stored biometric data.
- Explore deployment options for compatible edge devices.

