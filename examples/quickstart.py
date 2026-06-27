"""
FacePipe Quick Start Examples

Run these examples after installing: pip install facepipe
"""

import cv2
import numpy as np


def example_verify_two_faces():
    """Compare two face images — are they the same person?"""
    from facepipe.core.alignment.face_align import align_face
    from facepipe.core.detection.scrfd_detector import SCRFDDetector
    from facepipe.core.recognition.adaface_recognizer import AdaFaceRecognizer

    detector = SCRFDDetector()
    recognizer = AdaFaceRecognizer()

    # Load two images
    img1 = cv2.imread("photo1.jpg")
    img2 = cv2.imread("photo2.jpg")

    # Detect → Align → Extract embedding
    def get_embedding(img):
        faces = detector.detect(img)
        if not faces:
            raise ValueError("No face detected")
        aligned = align_face(img, faces[0].landmarks)
        return recognizer.extract(aligned).embedding

    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    # Compare (cosine similarity)
    similarity = float(np.dot(emb1, emb2))
    is_same = similarity > 0.4

    print(f"Similarity: {similarity:.4f}")
    print(f"Same person: {is_same}")


def example_quality_check():
    """Check if a face image is good enough for enrollment."""
    from facepipe.core.detection.scrfd_detector import SCRFDDetector
    from facepipe.core.quality.face_quality import FaceQualityAssessor

    detector = SCRFDDetector()
    quality = FaceQualityAssessor()

    img = cv2.imread("photo.jpg")
    faces = detector.detect(img)

    if not faces:
        print("No face detected")
        return

    report = quality.assess(img, faces[0])
    print(f"Quality Score:  {report.composite_score:.2f}")
    print(f"Blur Score:     {report.blur_score:.2f}")
    print(f"Pose (yaw):     {report.yaw:.1f}°")
    print(f"Pose (pitch):   {report.pitch:.1f}°")
    print(f"Good for enrollment: {report.passes_enrollment}")
    print(f"Good for recognition: {report.passes_recognition}")


def example_full_pipeline():
    """Use the full pipeline with all security checks."""
    from facepipe.core.pipeline import RecognitionPipeline

    pipeline = RecognitionPipeline()
    pipeline.initialize()

    img = cv2.imread("photo.jpg")

    # Enroll a new identity
    result = pipeline.enroll(
        frames=[img],
        identity_id="user_001",
        name="John Doe",
    )
    print(f"Enrolled: {result.success}, embeddings: {result.embeddings_stored}")

    # Recognize from a new image
    img2 = cv2.imread("photo2.jpg")
    frame_result = pipeline.process_frame(img2)

    for face in frame_result.faces:
        print(f"Identity: {face.decision.identity}")
        print(f"Confidence: {face.decision.confidence:.3f}")
        print(f"Recognized: {face.decision.is_recognized}")
        print(f"Liveness: {face.liveness.is_live}")
        print(f"Deepfake: {face.deepfake.is_real}")


def example_template_aggregation():
    """Aggregate multiple frames for better accuracy."""
    from facepipe.core.recognition.template_aggregator import TemplateAggregator

    aggregator = TemplateAggregator()

    # Simulate 5 embeddings with different quality scores
    embeddings = [np.random.randn(512).astype(np.float32) for _ in range(5)]
    for e in embeddings:
        e /= np.linalg.norm(e)

    quality_scores = [0.9, 0.7, 0.8, 0.3, 0.95]

    result = aggregator.aggregate(embeddings, quality_scores)
    print(f"Used {result.num_used}/{result.num_inputs} embeddings")
    print(f"Rejected {result.num_rejected} outliers")
    print(f"Strategy: {result.strategy}")


if __name__ == "__main__":
    print("=== FacePipe Quick Start ===\n")
    print("Available examples:")
    print("  1. example_verify_two_faces()   — Compare two face images")
    print("  2. example_quality_check()      — Check face quality")
    print("  3. example_full_pipeline()      — Full pipeline with enrollment")
    print("  4. example_template_aggregation() — Multi-frame aggregation")
    print("\nRun: python examples/quickstart.py")
