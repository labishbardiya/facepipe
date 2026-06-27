<div align="center">
  <img src="assets/logo.png" alt="FacePipe Logo" width="350" />
</div>

<div align="center">

**The production-ready, highly secure face recognition framework.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/labishbardiya/facepipe/actions/workflows/ci.yml/badge.svg)](https://github.com/labishbardiya/facepipe/actions)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-blue.svg)](https://labishbardiya.github.io/facepipe/)

</div>

---

**FacePipe** bridges the gap between academic face recognition and enterprise-grade deployment. It handles the heavy lifting of security, image quality, edge cases, and high-speed vector search so you can focus on building your application.

## 🌟 Key Features

* 🛡️ **Anti-Spoofing & Deepfake Detection:** Built-in defenses against presentation attacks and synthetic faces.
* 📈 **Test-Time Augmentation (TTA):** State-of-the-art accuracy through multi-pass flip averaging.
* 📸 **Quality Gating & Restoration:** Rejects bad captures, and automatically restores surveillance-quality faces using CodeFormer.
* 🧠 **Decision Fusion Engine:** Doesn't just blindly match embeddings—fuses 7 different signals (liveness, quality, tracking) into one secure verdict.
* ⚡ **Sub-Millisecond Search:** Integrated FAISS HNSW backend for blazing-fast 1:N retrieval.
* 🔒 **Encrypted Storage:** AES-256 encryption at rest for all biometric templates.

---

## ⚡ Quick Start

### 1. Install

```bash
pip install facepipe
```

### 2. Verify Two Faces

```python
import cv2
from facepipe import SCRFDDetector, AdaFaceRecognizer
from facepipe.core.alignment.face_align import align_face
import numpy as np

detector = SCRFDDetector()
recognizer = AdaFaceRecognizer()

def get_embedding(path):
    img = cv2.imread(path)
    face = detector.detect(img)[0]
    aligned = align_face(img, face.landmarks)
    return recognizer.extract(aligned).embedding

emb1 = get_embedding("alice1.jpg")
emb2 = get_embedding("alice2.jpg")

similarity = float(np.dot(emb1, emb2))
print(f"Match: {similarity > 0.4} (Similarity: {similarity:.4f})")
```

---

## 📖 Documentation

The example above just scratches the surface. For full tutorials on the **Decision Pipeline**, **Quality Assessment**, and **FAISS Vector Search**, visit our official documentation:

👉 **[Read the FacePipe Documentation](https://labishbardiya.github.io/facepipe/)**

- [Getting Started](https://labishbardiya.github.io/facepipe/getting-started/)
- [Architecture Deep-Dive](https://labishbardiya.github.io/facepipe/architecture/)
- [Security & Anti-Spoofing](https://labishbardiya.github.io/facepipe/security-and-anti-spoofing/)
- [API Reference](https://labishbardiya.github.io/facepipe/api/)

---

## 🛠️ Interactive Demo

FacePipe includes a built-in Gradio interface to help you visually test the quality and verification metrics.

```bash
pip install "facepipe[demo]"
facepipe demo
```
*Opens a local web interface at `http://localhost:7860`*

---

<div align="center">
  <i>Developed with ❤️ by <a href="https://github.com/labishbardiya">Labish Bardiya</a>. Licensed under MIT.</i>
</div>
