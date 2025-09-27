# MTCNN (Multi-task Cascaded Convolutional Networks)

MTCNN is a deep learning-based framework for **face detection** and **facial landmark localization**. It is widely used in computer vision projects for detecting faces in images and video streams.

---

## Features

- **Face Detection**: Detects bounding boxes around human faces in images or video.
- **Facial Landmark Detection**: Locates key facial points such as eyes, nose, and mouth corners.
- **High Accuracy**: Effective even with small, rotated, or partially occluded faces.
- **Handles Multiple Faces**: Detects multiple faces in a single image.
- **Robust to Lighting Conditions**: Works well in varying illumination scenarios.

---

## How It Works

MTCNN uses a **cascaded architecture** with three stages:

1. **P-Net (Proposal Network)**: Generates candidate face regions.
2. **R-Net (Refine Network)**: Refines candidate regions, removing false positives.
3. **O-Net (Output Network)**: Produces final bounding boxes and facial landmarks.

Each stage uses convolutional neural networks (CNNs) to progressively improve detection accuracy.

---

## Output Format

`detector.detect_faces(image)` returns a list of faces:

```python
[
    {
        'box': [x, y, w, h],          # Bounding box coordinates
        'confidence': 0.98,           # Detection confidence
        'keypoints': {                # Facial landmarks
            'left_eye': (x1, y1),
            'right_eye': (x2, y2),
            'nose': (x3, y3),
            'mouth_left': (x4, y4),
            'mouth_right': (x5, y5)
        }
    },
    ...
]
