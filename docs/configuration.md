# Configuration Reference

FacePipe uses `pydantic-settings` to manage configuration. All settings can be provided as environment variables prefixed with `FR_`.

## General Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FR_LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `FR_DATA_DIR` | `./data` | Path to store vectors and logs |
| `FR_MODELS_DIR` | `./models` | Path where downloaded models are cached |

## Quality Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `FR_QUALITY_BLUR_THRESHOLD` | `100.0` | Minimum Laplacian variance |
| `FR_QUALITY_ENROLLMENT_THRESHOLD`| `0.65` | Minimum composite score to enroll a new face |
| `FR_QUALITY_RECOGNITION_THRESHOLD`| `0.45` | Minimum score to attempt recognition |

## Recognition & Security

| Variable | Default | Description |
|----------|---------|-------------|
| `FR_RECOGNITION_MODEL` | `adaface` | `adaface` or `arcface` |
| `FR_RECOGNITION_TTA_ENABLED` | `true` | Enable flip-averaging augmentation |
| `FR_FUSION_SECURITY_LEVEL` | `STANDARD`| `STANDARD`, `ELEVATED`, or `MAXIMUM` |
| `FR_DEEPFAKE_ENABLED` | `true` | Enable deepfake detector |

*See `.env.example` in the root repository for the complete list of 40+ configuration options.*
