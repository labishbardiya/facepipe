"""
FacePipe Interactive Demo — Gradio UI

Launch: python demo/app.py
Then open: http://localhost:7860

Features:
  - Face Verification: Upload two photos, see if they're the same person
  - Quality Assessment: Upload a photo, see detailed quality metrics
  - Face Detection: Upload a photo, see detected faces with landmarks
"""

from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    print("Or: pip install facepipe[demo]")
    sys.exit(1)

from facepipe.core.detection.scrfd_detector import SCRFDDetector
from facepipe.core.alignment.face_align import align_face
from facepipe.core.recognition.adaface_recognizer import AdaFaceRecognizer
from facepipe.core.quality.face_quality import FaceQualityAssessor

# Initialize models (lazy, loaded on first use)
detector = SCRFDDetector()
recognizer = AdaFaceRecognizer()
quality_assessor = FaceQualityAssessor()


def _detect_and_embed(img_bgr: np.ndarray):
    """Detect face, align, extract embedding. Returns (embedding, face, quality)."""
    faces = detector.detect(img_bgr)
    if not faces:
        return None, None, None

    face = faces[0]
    quality_report = quality_assessor.assess(img_bgr, face)
    aligned = align_face(img_bgr, face.landmarks)
    emb_result = recognizer.extract(aligned)

    return emb_result.embedding, face, quality_report


def verify_faces(img1, img2):
    """Compare two face images and return similarity analysis."""
    if img1 is None or img2 is None:
        return "Please upload both images."

    # Convert RGB (Gradio) to BGR (OpenCV)
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    emb1, face1, q1 = _detect_and_embed(img1_bgr)
    emb2, face2, q2 = _detect_and_embed(img2_bgr)

    if emb1 is None:
        return "❌ No face detected in Image 1"
    if emb2 is None:
        return "❌ No face detected in Image 2"

    similarity = float(np.dot(emb1, emb2))

    # Decision
    if similarity > 0.5:
        verdict = "✅ SAME PERSON"
        confidence = "High"
    elif similarity > 0.4:
        verdict = "⚠️ POSSIBLY SAME PERSON"
        confidence = "Medium"
    else:
        verdict = "❌ DIFFERENT PEOPLE"
        confidence = "High"

    result = f"""## {verdict}

| Metric | Value |
|--------|-------|
| **Cosine Similarity** | {similarity:.4f} |
| **Confidence** | {confidence} |
| **Threshold** | 0.40 (standard) |

### Image 1 Quality
| Metric | Score |
|--------|-------|
| Composite | {q1.composite_score:.2f} |
| Blur | {q1.blur_score:.2f} |
| Pose (yaw/pitch) | {q1.yaw:.1f}° / {q1.pitch:.1f}° |

### Image 2 Quality
| Metric | Score |
|--------|-------|
| Composite | {q2.composite_score:.2f} |
| Blur | {q2.blur_score:.2f} |
| Pose (yaw/pitch) | {q2.yaw:.1f}° / {q2.pitch:.1f}° |
"""
    return result


def assess_quality(img):
    """Assess face quality for a single image."""
    if img is None:
        return "Please upload an image.", img

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = detector.detect(img_bgr)

    if not faces:
        return "❌ No face detected in image.", img

    # Annotate image with bounding boxes
    annotated = img.copy()
    results = []

    for i, face in enumerate(faces):
        quality = quality_assessor.assess(img_bgr, face)

        # Draw bounding box
        x1, y1, x2, y2 = face.bbox.astype(int)
        color = (0, 200, 0) if quality.passes_enrollment else (200, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, f"Q:{quality.composite_score:.2f}",
            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        # Draw landmarks
        if face.landmarks is not None:
            for lm in face.landmarks:
                cv2.circle(annotated, (int(lm[0]), int(lm[1])), 2, (0, 255, 0), -1)

        status = "✅ Good" if quality.passes_enrollment else "⚠️ Low Quality"
        results.append(f"""### Face {i + 1} — {status}

| Metric | Value | Status |
|--------|-------|--------|
| **Composite Score** | {quality.composite_score:.3f} | {'✅' if quality.composite_score > 0.65 else '⚠️'} |
| **Blur** | {quality.blur_score:.1f} | {'✅' if quality.blur_score > 100 else '⚠️ Blurry'} |
| **Yaw** | {quality.yaw:.1f}° | {'✅' if abs(quality.yaw) < 30 else '⚠️ Turned'} |
| **Pitch** | {quality.pitch:.1f}° | {'✅' if abs(quality.pitch) < 25 else '⚠️ Tilted'} |
| **Detection Conf** | {face.score:.3f} | ✅ |
| **Enrollment OK** | {quality.passes_enrollment} | {'✅' if quality.passes_enrollment else '❌'} |
""")

    return "\n".join(results), annotated


# ── Build Gradio UI ──────────────────────────────────────────

with gr.Blocks(
    title="FacePipe — Face Recognition Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
    # 🔍 FacePipe — Face Recognition Demo

    Production-ready face recognition pipeline with quality assessment,
    anti-spoofing, deepfake detection, and decision fusion.

    **[GitHub](https://github.com/labishbardiya/facepipe)** |
    **[Docs](https://github.com/labishbardiya/facepipe#readme)** |
    `pip install facepipe`
    """)

    with gr.Tabs():
        # ── Tab 1: Face Verification ─────────────────────────
        with gr.TabItem("🔐 Face Verification"):
            gr.Markdown("Upload two face images to check if they belong to the same person.")
            with gr.Row():
                img1_input = gr.Image(label="Image 1", type="numpy")
                img2_input = gr.Image(label="Image 2", type="numpy")
            verify_btn = gr.Button("Compare Faces", variant="primary")
            verify_output = gr.Markdown(label="Result")
            verify_btn.click(verify_faces, inputs=[img1_input, img2_input], outputs=verify_output)

        # ── Tab 2: Quality Assessment ────────────────────────
        with gr.TabItem("📊 Quality Assessment"):
            gr.Markdown("Upload a face image to see detailed quality metrics and enrollment eligibility.")
            quality_input = gr.Image(label="Upload Image", type="numpy")
            quality_btn = gr.Button("Assess Quality", variant="primary")
            quality_text = gr.Markdown(label="Quality Report")
            quality_annotated = gr.Image(label="Annotated Image")
            quality_btn.click(assess_quality, inputs=quality_input, outputs=[quality_text, quality_annotated])

    gr.Markdown("""
    ---
    *Powered by FacePipe — MIT Licensed | Models auto-download on first use*
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
