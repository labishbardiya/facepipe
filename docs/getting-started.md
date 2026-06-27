# Getting Started

This guide will walk you through installing FacePipe, running your first verification check, and deploying the interactive demo.

## Installation

FacePipe requires Python 3.10+. We highly recommend installing it in a virtual environment.

```bash
pip install facepipe
```

### Models
FacePipe automatically downloads the necessary InsightFace models (SCRFD and ArcFace) to your `~/.insightface/` directory the first time you run it. 

If you want to use the premium AdaFace or CodeFormer models, you can run the CLI to get download instructions:

```bash
facepipe models
```

---

## 1. Quick Verification (The Basics)

If you just want to see if two photos are of the same person, you can use the core components directly.

```python title="verify.py"
import cv2
import numpy as np
from facepipe import SCRFDDetector, AdaFaceRecognizer
from facepipe.core.alignment.face_align import align_face

detector = SCRFDDetector()
recognizer = AdaFaceRecognizer()

def get_embedding(path):
    img = cv2.imread(path)
    faces = detector.detect(img)
    if not faces:
        raise ValueError("No face detected!")
    
    # Align and extract
    aligned = align_face(img, faces[0].landmarks)
    return recognizer.extract(aligned).embedding

emb1 = get_embedding("photo1.jpg")
emb2 = get_embedding("photo2.jpg")

# Cosine similarity
similarity = float(np.dot(emb1, emb2))
print(f"Similarity: {similarity:.4f}")
print(f"Same Person: {similarity > 0.40}")
```

---

## 2. The Production Pipeline

In production, you don't just want similarity. You want to block deepfakes, reject blurry photos, and track identities securely. The `RecognitionPipeline` orchestrates all of this.

```python title="pipeline.py"
import cv2
from facepipe import RecognitionPipeline

# Initialize the pipeline (loads all models, connects to database)
pipeline = RecognitionPipeline()
pipeline.initialize()

# 1. Enroll a user
enrollment_img = cv2.imread("alice_id.jpg")
enroll_result = pipeline.enroll(
    frames=[enrollment_img],
    identity_id="usr_123",
    name="Alice"
)
print(f"Enrollment Success: {enroll_result.success}")

# 2. Recognize from a camera frame
webcam_img = cv2.imread("webcam_frame.jpg")
result = pipeline.process_frame(webcam_img)

for face in result.faces:
    print(f"Identity: {face.decision.identity}")
    print(f"Recognized: {face.decision.is_recognized}")
    
    # Check security metrics!
    print(f"Deepfake detected: {not face.deepfake.is_real}")
    print(f"Spoof detected: {not face.liveness.is_live}")
    print(f"Quality Score: {face.quality.composite_score:.2f}")
```

---

## 3. Interactive Demo

FacePipe includes a built-in Gradio interface to help you visually test the quality and verification metrics.

First, ensure you have the demo dependencies:
```bash
pip install facepipe[demo]
```

Then run the demo:
```bash
facepipe demo
```
This will open a browser window at `http://localhost:7860` with a full graphical interface.
