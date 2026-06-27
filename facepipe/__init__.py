"""
FacePipe — Production-ready face recognition pipeline.
"""

from facepipe.core.detection.scrfd_detector import SCRFDDetector
from facepipe.core.models import check_models
from facepipe.core.pipeline import RecognitionPipeline
from facepipe.core.quality.face_quality import FaceQualityAssessor
from facepipe.core.recognition.adaface_recognizer import AdaFaceRecognizer
from facepipe.core.recognition.ensemble_recognizer import EnsembleRecognizer

__version__ = "2.1.0"
__all__ = [
    "RecognitionPipeline",
    "SCRFDDetector",
    "AdaFaceRecognizer",
    "EnsembleRecognizer",
    "FaceQualityAssessor",
    "check_models",
]
