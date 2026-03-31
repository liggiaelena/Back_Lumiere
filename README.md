# Skin Analyzer Backend

Backend API for facial skin analysis using MediaPipe for face detection and Claude Vision for per-region skin assessment.

## Stack

- **FastAPI** + **Uvicorn** — REST API server
- **MediaPipe** — facial landmark detection and region segmentation
- **OpenCV** + **Pillow** — image processing and validation
- **Anthropic SDK** — Claude Vision for skin analysis per facial region

## How It Works

1. Image is uploaded via `POST /api/analyze`
2. Image is validated (size, brightness, format) and preprocessed
3. MediaPipe detects 5 facial regions: forehead, cheeks (L/R), nose, chin
4. Each region is cropped and sent concurrently to Claude Vision
5. Claude returns structured JSON per region (tone, oiliness, imperfections, etc.)
6. Results are aggregated into a final report with cross-region color comparison

## Project Structure

```
backend-skin-analyzer/
├── app/
│   ├── main.py             # FastAPI app, routes, CORS
│   ├── pipeline.py         # Orchestrates the full analysis flow
│   ├── vision.py           # Claude Vision calls (async, per region)
│   ├── mediapipe_utils.py  # Landmark detection + region cropping
│   ├── image_utils.py      # Load, validate, preprocess image
│   ├── color_utils.py      # Color delta calc + final report builder
│   └── config.py           # Settings via pydantic-settings + .env
├── .env
├── requirements.txt
└── run.py
```

## Setup

### 1. Clone and install dependencies

```bash
cd backend-skin-analyzer
pip install -r requirements.txt
```

### 2. Configure environment

Edit `.env` and add your Anthropic API key:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run the server

```bash
python run.py
```

Server starts at `http://localhost:8000` with hot reload enabled.

## API

### `GET /`

Health check.

```json
{ "status": "ok", "service": "skin-analyzer" }
```

### `POST /api/analyze`

Analyzes a facial photo.

**Request:** `multipart/form-data` with a `file` field (JPG, PNG, or WebP, max 10MB).

**Response:**

```json
{
  "tom_geral_fitzpatrick": 3,
  "subtom_predominante": "quente",
  "regioes": {
    "testa": {
      "tom_hex": "#C8916A",
      "tom_fitzpatrick": 3,
      "subtom": "quente",
      "oleosidade": "misto",
      "imperfeicoes": [
        { "tipo": "poro", "intensidade": "leve" }
      ],
      "uniformidade": 7,
      "notas": "Pele com leve irregularidade de textura."
    },
    "bochecha_e": { "..." : "..." },
    "bochecha_d": { "..." : "..." },
    "nariz":      { "..." : "..." },
    "queixo":     { "..." : "..." }
  },
  "comparacao_tons": {
    "testa_vs_bochecha_e": { "delta": 12.4, "nivel": "moderado" },
    "testa_vs_bochecha_d": { "delta": 8.1,  "nivel": "baixo" },
    "nariz_vs_bochecha_e": { "delta": 21.0, "nivel": "alto" },
    "queixo_vs_testa":     { "delta": 5.3,  "nivel": "baixo" }
  },
  "imperfeicoes": [
    { "tipo": "poro", "intensidade": "leve", "regiao": "testa" }
  ]
}
```

**Error responses:**

| Status | Reason |
|--------|--------|
| 400 | Invalid format or file too large |
| 422 | Image too small, too dark, too bright, or no face detected |
| 500 | Internal server error |

## Image Requirements

- Format: JPG, PNG, or WebP
- Minimum size: 300×300px
- Maximum size: 10MB
- Face must be centered and well-lit
- Avoid extreme overexposure or very dark images

## CORS

Allowed origins by default: `http://localhost:5173` and `http://localhost:3000`. Update `main.py` to add production origins.
