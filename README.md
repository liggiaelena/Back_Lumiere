# Skin Analyzer Backend

FastAPI backend for facial skin analysis using MediaPipe for face detection, BiSeNet for skin-tone extraction, and Claude Vision for per-region skin assessment.

## Stack

- FastAPI + Uvicorn: REST API server
- MediaPipe Tasks API: facial landmark detection and region segmentation
- BiSeNet + PyTorch: skin mask extraction and RGB estimation
- OpenCV + Pillow: image processing and validation
- Anthropic SDK: Claude Vision skin analysis per facial region

## How It Works

1. Image is uploaded via `POST /api/analyze`.
2. Image is validated and preprocessed.
3. MediaPipe detects facial regions: forehead, cheeks, nose, and chin.
4. BiSeNet estimates the global skin RGB/HEX value.
5. Region crops are sent concurrently to Claude Vision.
6. Results are aggregated into a final report with color comparison and foundation recommendations.

## Project Structure

```text
Back_Lumiere/
├── README.md
├── requirements.txt         # project dependencies
├── .env                     # API key, never commit this
├── data-collection/
│   ├── outputs/             # Local intermediate images and JSON outputs
│   └── scripts/             # Face parsing, skin extraction, RGB estimation
├── dev/
│   ├── app/
│   │   ├── main.py          # FastAPI app, routes, CORS
│   │   ├── pipeline.py      # Orchestrates the full analysis flow
│   │   ├── vision.py        # Claude Vision / provider integration
│   │   ├── mediapipe_utils.py
│   │   ├── image_utils.py
│   │   ├── color_utils.py
│   │   ├── recommendations.py
│   │   ├── skin_tone_analyzer.py
│   │   └── config.py
│   └── run.py               # Development server entrypoint
├── documentation/
│   └── US160_SegFormer.md   # Technical evaluation and hardware benchmark report for SegFormer selection
└── training/
    ├── checkpoints/         # BiSeNet weights
    ├── SegFormer/
│   │   └── benchmark.py     # Performance benchmarking script for SegFormer variants (B0-B5)
    └── models/              # BiSeNet model architecture
    
```

## Setup

```bash
cd Back_Lumiere
python -m venv venv
venv\Scripts\Activate.ps1   # (PowerShell)
# or: venv\Scripts\activate  # (Git Bash / CMD)
pip install -r requirements.txt
```

Create `.env` in `Back_Lumiere/` (example):

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
# or GEMINI_API_KEY=your-gemini-key-here
```

## Run Development API

Recommended (from project root so `.env` is loaded):

```powershell
cd L:\Lumiere\Back_Lumiere
venv\Scripts\python dev\run.py
```

Alternative (run from `dev/`):

```powershell
cd dev
..\venv\Scripts\python run.py
```

Server starts at `http://localhost:8001`.

## Data Collection Scripts

Place `face_crop.jpg` in `data-collection/outputs/`, then run:

```bash
python data-collection/scripts/face_parsing_bisenet.py
python data-collection/scripts/skin_region_extraction.py
python data-collection/scripts/rgb_estimation.py
```

Generated files are written to `data-collection/outputs/`.

## API

### `GET /`

Health check.

```json
{ "status": "ok", "service": "skin-analyzer" }
```

### `POST /api/analyze`

Analyzes a facial photo.

Request: `multipart/form-data` with a `file` field. Supported formats are JPG, PNG, and WebP. Maximum size is 10 MB.

## Image Requirements

- Format: JPG, PNG, or WebP
- Maximum size: 10 MB
- Face should be centered and well-lit

## Python Version

Requires Python 3.13.
