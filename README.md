# Skin Analyzer — Backend

FastAPI backend for facial skin analysis using MediaPipe for face detection and Claude Vision for per-region skin assessment.

## Stack

- **FastAPI** + **Uvicorn** — REST API server
- **MediaPipe** (Tasks API) — facial landmark detection and region segmentation
- **OpenCV** + **Pillow** — image processing and validation
- **Anthropic SDK** — Claude Vision for skin analysis per facial region

## How It Works

1. Image is uploaded via `POST /api/analyze`
2. Image is validated (size, format) and preprocessed
3. MediaPipe detects 5 facial regions: forehead, cheeks (L/R), nose, chin
4. Each region is cropped and sent concurrently to Claude Vision
5. Claude returns structured JSON per region (tone, oiliness, imperfections, etc.)
6. Results are aggregated into a final report with cross-region color comparison and foundation recommendations

## Project Structure

```
Back_Lumiere/
├── app/
│   ├── main.py              # FastAPI app, routes, CORS
│   ├── pipeline.py          # Orchestrates the full analysis flow
│   ├── vision.py            # Claude Vision calls (async, per region)
│   ├── mediapipe_utils.py   # Landmark detection + region cropping
│   ├── image_utils.py       # Load, validate, preprocess image
│   ├── color_utils.py       # Color delta calc + final report builder
│   ├── recommendations.py   # Foundation shade database + matching logic
│   └── config.py            # Settings via pydantic-settings + .env
├── .env                     # API key — never commit this
├── requirements.txt
└── run.py
```

## Setup

### 1. Clone and create virtual environment

```bash
cd Back_Lumiere
python -m venv venv
venv/Scripts/activate       # Windows
# source venv/bin/activate  # Mac/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> On first request, the app will automatically download the MediaPipe face landmark model (~3.5MB) to your home directory.

### 3. Configure environment

Create a `.env` file in the `Back_Lumiere/` folder:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 4. Run the server

```bash
venv/Scripts/python run.py
```

Server starts at `http://localhost:8001`.

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
  "tom_geral_fitzpatrick": 4,
  "subtom_predominante": "quente",
  "regioes": {
    "testa": {
      "tom_hex": "#C8916A",
      "tom_fitzpatrick": 4,
      "subtom": "quente",
      "oleosidade": "normal",
      "imperfeicoes": [
        { "tipo": "poro", "intensidade": "leve" }
      ],
      "uniformidade": 7,
      "notas": "Uniform tone with slight texture irregularity."
    },
    "bochecha_e": { "...": "..." },
    "bochecha_d": { "...": "..." },
    "nariz":      { "...": "..." },
    "queixo":     { "...": "..." }
  },
  "comparacao_tons": {
    "testa_vs_bochecha_e": { "delta": 12.4, "nivel": "moderado" },
    "testa_vs_bochecha_d": { "delta": 8.1,  "nivel": "baixo" },
    "nariz_vs_bochecha_e": { "delta": 21.0, "nivel": "alto" },
    "queixo_vs_testa":     { "delta": 5.3,  "nivel": "baixo" }
  },
  "imperfeicoes": [
    { "tipo": "poro", "intensidade": "leve", "regiao": "testa" }
  ],
  "recommendations": [
    {
      "brand": "Fenty Beauty",
      "shade_name": "340W",
      "shade_code": "340W",
      "undertone": "quente",
      "fitzpatrick_range": [3, 4],
      "price_range": "$38–$42",
      "where_to_buy": "https://www.fentybeauty.com"
    }
  ]
}
```

**Error responses:**

| Status | Reason |
|--------|--------|
| 400 | Invalid format or file too large |
| 422 | No face detected in image |
| 500 | Internal server error |

## Image Requirements

- Format: JPG, PNG, or WebP
- Maximum size: 10MB
- Face must be centered and well-lit

## CORS

Accepts requests from any `localhost` port (all `http://localhost:*` origins are allowed). Update `main.py` to restrict in production.

## Python Version

Requires **Python 3.13**. The `requirements.txt` pins versions compatible with Python 3.13.
