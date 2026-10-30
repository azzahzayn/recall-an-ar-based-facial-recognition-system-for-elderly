from setuptools import setup, find_packages

setup(
    name="face_ar_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "mediapipe",
        "onnxruntime",
        "pyyaml",
        "pydantic",
        "scipy",
        "pytest",
        "pillow"
    ],
)
