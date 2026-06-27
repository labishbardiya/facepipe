# Contributing to FacePipe

Thank you for considering contributing to FacePipe! This document explains how to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/labishbardiya/facepipe.git
cd facepipe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=evaluation --cov-report=html

# Run a specific test file
pytest tests/test_core.py -v
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix
ruff check --fix .

# Format
ruff format .
```

## Project Structure

```
facepipe/
├── api/              # FastAPI REST endpoints
├── config/           # Pydantic settings (single source of truth)
├── core/             # Pipeline components
│   ├── alignment/    # Face alignment (similarity transform)
│   ├── antispoof/    # Liveness detection
│   ├── clustering/   # Per-identity appearance clustering
│   ├── deepfake/     # Deepfake detection (multi-signal)
│   ├── detection/    # SCRFD face detection
│   ├── fusion/       # Decision fusion engine
│   ├── learning/     # Active learning gate
│   ├── quality/      # Quality assessment + face restoration
│   ├── recognition/  # AdaFace/ArcFace + template aggregation + ensemble
│   ├── search/       # FAISS HNSW + score normalization
│   └── tracking/     # ByteTrack temporal tracking
├── demo/             # Gradio interactive demo
├── entrypoints/      # CLI + server entry points
├── evaluation/       # Benchmark harness + failure analysis
├── examples/         # Quick-start examples
├── observability/    # Structured logging + Prometheus metrics
├── storage/          # Encrypted storage + event sourcing
└── tests/            # Unit tests
```

## Adding a New Pipeline Component

1. Create your module in the appropriate `core/` subdirectory
2. Add configuration to `config/settings.py` as a new `BaseSettings` subclass
3. Register the settings in the root `Settings` class
4. Wire the component into `core/pipeline.py`
5. Add unit tests in `tests/`
6. Update the README if it's user-facing

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add/update tests for any new functionality
4. Ensure `pytest` passes and `ruff check .` is clean
5. Update documentation if needed
6. Submit a PR with a description of what and why

## Reporting Issues

- Use the [GitHub Issues](https://github.com/labishbardiya/facepipe/issues) page
- Include: Python version, OS, error traceback, and steps to reproduce
- For performance issues, include hardware specs (CPU/GPU)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
