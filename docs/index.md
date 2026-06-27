# FacePipe

**The production-ready face recognition pipeline.**

FacePipe is a complete, modular, and encrypted face recognition framework built for production environments. It goes beyond simple similarity matching by integrating state-of-the-art security, quality, and accuracy enhancements.

---

## Why FacePipe?

While tools like `face_recognition` or `DeepFace` are great for prototyping, putting them into production requires manually building pipelines for spoofing, bad lighting, deepfakes, and multi-frame video tracking. FacePipe solves this out of the box.

### Features

- 🕵️ **State-of-the-Art Recognition:** Uses ArcFace/AdaFace with Test-Time Augmentation (TTA).
- 🛡️ **Anti-Spoofing & Liveness:** Prevents presentation attacks.
- 🎭 **Deepfake Detection:** Flags synthetic faces automatically.
- 📊 **Quality Assessment:** Rejects blurry, poorly lit, or occluded faces.
- 🔧 **Face Restoration:** Restores low-quality surveillance faces using CodeFormer.
- 🧮 **Decision Fusion Engine:** Combines 7 different signals (quality, liveness, deepfake, tracking) to make secure decisions.
- 🚀 **Scalable Search:** Uses FAISS HNSW for sub-millisecond vector search.
- 🔒 **Encrypted Storage:** AES-256 encryption for all biometric templates at rest.

## How it works

FacePipe uses a multi-stage pipeline. Check out the [Architecture](architecture.md) page for a deep dive.

```mermaid
graph LR
    A[Image] --> B[SCRFD Detect]
    B --> C[Quality Gate]
    C --> D[Liveness / Deepfake]
    D --> E[Face Alignment]
    E --> F[AdaFace Extract]
    F --> G[Score Normalization]
    G --> H[Decision Fusion]
```

## Ready to dive in?

Get started quickly with our [Getting Started](getting-started.md) guide, or explore the [API Reference](api.md) for detailed integration instructions.
