"""
Real-time webcam HUD for facial recognition and enrollment.
"""

import argparse
import sys
import time

import cv2
from core.pipeline import RecognitionPipeline
from storage.event_store import EventStore, EventType
from storage.identity_manager import IdentityManager


def main():
    parser = argparse.ArgumentParser(description="Real-time Facial Recognition HUD")
    parser.add_argument("--enroll", type=str, help="Name to enroll from the webcam")
    args = parser.parse_args()

    print("Initializing Pipeline (loading models, please wait)...")
    pipeline = RecognitionPipeline()
    pipeline.initialize()

    identity_mgr = IdentityManager()
    event_store = EventStore()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        sys.exit(1)

    enrollment_frames = []
    is_enrolling = args.enroll is not None
    enrollment_name = args.enroll

    if is_enrolling:
        print(f"\n--- ENROLLMENT MODE for '{enrollment_name}' ---")
        print("Please look at the camera. We will capture 3 high-quality frames.")
    else:
        print("\n--- RECOGNITION MODE ---")

    print("\nPress 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------------------------------------------------
        # ENROLLMENT MODE
        # ---------------------------------------------------------
        if is_enrolling:
            faces = pipeline._detector.detect(frame)
            if len(faces) == 1:
                face = faces[0]
                quality = pipeline._quality.assess(face.crop, face.landmarks, face.frame_area_ratio)

                # Determine bounding box color based on quality
                x1, y1, x2, y2 = face.bbox.astype(int)
                color = (0, 255, 0) if quality.passes_enrollment else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Show status
                status = "Quality: Good" if quality.passes_enrollment else f"Poor Quality: {', '.join(quality.rejection_reasons)}"
                cv2.putText(frame, f"Capturing: {len(enrollment_frames)}/3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Capture frame if quality is good
                if quality.passes_enrollment:
                    enrollment_frames.append(frame.copy())
                    cv2.imshow("Real-time HUD", frame)
                    cv2.waitKey(1)
                    time.sleep(0.5) # small delay to get slightly varied angles

                # Once we have 3 frames, process the enrollment
                if len(enrollment_frames) >= 3:
                    print("\nProcessing enrollment... please wait.")
                    cv2.putText(frame, "Processing...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow("Real-time HUD", frame)
                    cv2.waitKey(1)

                    result = pipeline.enroll(name=enrollment_name, frames=enrollment_frames)
                    if result.success:
                        # Save identity to database
                        identity_mgr.create(
                            name=result.name,
                            embedding_count=result.embeddings_stored,
                            model_version=pipeline._recognizer.model_version,
                            identity_id=result.identity_id,
                        )
                        # Log event
                        event_store.append(
                            EventType.IDENTITY_ENROLLED,
                            identity_id=result.identity_id,
                            payload={"name": result.name, "embeddings": result.embeddings_stored, "source": "webcam"}
                        )
                        print(f"✅ Successfully enrolled '{result.name}'!")
                    else:
                        print(f"❌ Enrollment failed: {result.message}")

                    is_enrolling = False
                    print("\n--- Switching to RECOGNITION MODE ---")
            else:
                cv2.putText(frame, "Please ensure exactly 1 face is visible", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # ---------------------------------------------------------
        # RECOGNITION MODE
        # ---------------------------------------------------------
        else:
            results = pipeline.process_frame(frame)

            for r in results:
                x1, y1, x2, y2 = r.bbox.astype(int)

                # Determine box color and label based on pipeline decisions
                if not r.liveness.is_live or not r.deepfake.is_real:
                    color = (0, 0, 255) # Red = Spoof / Fake
                    label = "SPOOF / FAKE"
                elif r.decision.is_recognized:
                    # Resolve name from identity ID
                    identity_record = identity_mgr.get(r.identity)
                    name = identity_record.name if identity_record else r.identity
                    color = (0, 255, 0) # Green = Recognized
                    label = f"{name} ({r.decision.confidence:.2f})"
                elif r.openset.decision == "ambiguous":
                    color = (0, 165, 255) # Orange = Ambiguous
                    label = "AMBIGUOUS"
                else:
                    color = (255, 0, 0) # Blue = Unknown
                    label = "UNKNOWN"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Name tag background and text
                cv2.rectangle(frame, (x1, max(0, y1 - 35)), (x2, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Extra details below box (Latency, Quality)
                details = f"Lat: {r.latency_ms:.0f}ms | Qual: {r.quality.composite_score:.2f}"
                cv2.putText(frame, details, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Real-time HUD", frame)

        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
