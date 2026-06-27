"""
Model auto-download and management.

Downloads required models on first use. Models are cached in the
configured models directory (default: ./models/).

Supported models:
  - SCRFD (face detection) — auto-downloaded via InsightFace buffalo_l
  - ArcFace R100 (recognition) — auto-downloaded via InsightFace buffalo_l
  - AdaFace IR101 (recognition) — manual download from HuggingFace
  - CodeFormer (face restoration) — optional, manual download
"""

from __future__ import annotations

import os
from pathlib import Path

from facepipe.config.settings import get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)

# Models that auto-download via InsightFace
INSIGHTFACE_MODELS = {
    "buffalo_l": {
        "description": "SCRFD detection + ArcFace R100 recognition bundle",
        "auto_download": True,
        "components": ["det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx", "2d106det.onnx", "genderage.onnx"],
    }
}

# Models that require manual download
MANUAL_MODELS = {
    "adaface_ir101": {
        "description": "AdaFace IR101 trained on WebFace4M — best for noisy/surveillance inputs",
        "filename": "adaface_ir101.onnx",
        "source": "https://huggingface.co/minchul/adaface_ir101_webface4m",
        "instructions": (
            "Download from HuggingFace:\n"
            "  pip install huggingface_hub\n"
            "  python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download('minchul/adaface_ir101_webface4m', "
            "'adaface_ir101_webface4m.onnx', local_dir='models/')\""
        ),
    },
    "codeformer": {
        "description": "CodeFormer face restoration — better identity preservation than GFPGAN",
        "filename": "codeformer.onnx",
        "source": "https://github.com/sczhou/CodeFormer/releases",
        "instructions": (
            "Download from CodeFormer releases:\n"
            "  https://github.com/sczhou/CodeFormer/releases\n"
            "  Place codeformer.onnx in models/ directory"
        ),
    },
}


def check_models() -> dict[str, dict]:
    """Check which models are available and which need downloading.

    Returns:
        Dict mapping model_name → {"available": bool, "path": str, "instructions": str}
    """
    settings = get_settings()
    models_dir = Path(settings.models_dir)
    insightface_dir = Path.home() / ".insightface" / "models" / "buffalo_l"

    status = {}

    # Check InsightFace models (auto-downloaded to ~/.insightface/)
    for name, info in INSIGHTFACE_MODELS.items():
        all_present = all(
            (insightface_dir / comp).exists() for comp in info["components"]
        )
        status[name] = {
            "available": all_present,
            "path": str(insightface_dir),
            "auto_download": True,
            "description": info["description"],
            "instructions": "Auto-downloads on first use via InsightFace" if not all_present else "Ready",
        }

    # Check manual models
    for name, info in MANUAL_MODELS.items():
        model_path = models_dir / info["filename"]
        status[name] = {
            "available": model_path.exists(),
            "path": str(model_path),
            "auto_download": False,
            "description": info["description"],
            "instructions": info["instructions"] if not model_path.exists() else "Ready",
        }

    return status


def print_model_status() -> None:
    """Print a formatted status table of all models."""
    status = check_models()

    print("=" * 70)
    print("MODEL STATUS")
    print("=" * 70)

    for name, info in status.items():
        icon = "✅" if info["available"] else "❌"
        auto = " (auto)" if info.get("auto_download") else ""
        print(f"\n  {icon} {name}{auto}")
        print(f"     {info['description']}")
        print(f"     Path: {info['path']}")
        if not info["available"]:
            print(f"     Instructions: {info['instructions']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_model_status()
