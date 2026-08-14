# API Contract

## 1. Overview

This document describes the current FastAPI backend in `Back_Lumiere/dev/app/main.py` for the Lumiere skin analyzer service.

- Backend framework: FastAPI
- Main route file: `Back_Lumiere/dev/app/main.py`
- Database persistence: `Back_Lumiere/dev/app/data_service.py`
- Frontend call site: `Front_Lumiere/src/services/api.js`

## 2. GET /

- Description: Service health check (服務健康檢查)
- Method: `GET`
- Path: `/`
- Response (200):
  ```json
  {
    "status": "ok",
    "service": "skin-analyzer",
    "db": "ok"
  }
  ```

- Notes:
  - Performs a simple DB connectivity check by executing `SELECT 1` against the configured engine.
  - Returns `db: "unreachable"` if the database connection fails.

## 3. POST /api/analyze

- Description: Upload a facial photo for skin analysis (上傳人臉照片並進行皮膚分析)
- Method: `POST`
- Path: `/api/analyze`
- Content-Type: `multipart/form-data`
- Request Fields:
  - `file` (required): Image binary content (圖檔二進位內容)
- Supported Formats (支援格式):
  - `image/jpeg`
  - `image/png`
  - `image/webp`
- Maximum File Size: 10 MB

### Possible Responses (可能回應)

- `200 OK`
  - `application/json`
  - Content: Analysis result object with `id` appended after persistence
- `400 Bad Request`
  - Reason: Unsupported file format (檔案格式不支援)
  - Example:
    ```json
    { "detail": "Invalid format. Use JPG, PNG or WebP." }
    ```
- `413 Payload Too Large`
  - Reason: File exceeds 10 MB (檔案超過 10 MB)
- `422 Unprocessable Entity`
  - Reason: Image content or validation failed (檔案內容或影像驗證失敗)
- `500 Internal Server Error`
  - Reason: Internal error while analyzing the image (分析時發生內部錯誤)

### Behavior

- Validates upload content type against `ALLOWED = {"image/jpeg", "image/png", "image/webp"}`.
- Reads the full file into memory and rejects if larger than `10 * 1024 * 1024` bytes.
- Uses `app.image_utils.load_and_validate()` to validate the image bytes.
- Calls `app.pipeline.run_pipeline()` to generate the analysis result.
- Persists the result through `app.data_service.save_analysis()` and returns the saved `id`.

## 4. GET /api/analyze/{analyze_id}

- Description: Retrieve a saved analysis result by ID (依分析 ID 取得已保存的分析結果)
- Method: `GET`
- Path: `/api/analyze/{analyze_id}`
- Path Parameters:
  - `analyze_id` (string): Analysis record identifier

### Possible Responses

- `200 OK`
  - `application/json`
  - Content: Saved analysis object including `id` and `created_at`
- `404 Not Found`
  - Reason: Analysis result not found for the given ID (分析結果不存在)
- `500 Internal Server Error`
  - Reason: Internal error while fetching the analysis (取得分析結果時發生內部錯誤)

### Behavior

- Calls `app.data_service.get_analysis(analyze_id)`.
- If the row exists, it returns the JSON result plus `id` and `created_at`.
- If no row is found, returns `404`.

## 5. Frontend API Usage (前端 API 用法)

- Frontend base URL: `http://localhost:8001` (default) (前端呼叫位址；預設)
- Frontend service file: `Front_Lumiere/src/services/api.js`
- Wrapper function:
  - `analyzeImage(file)`
  - Sends `multipart/form-data` to `/api/analyze`

## 6. CORS and Access

- CORS is enabled in `Back_Lumiere/dev/app/main.py` via `CORSMiddleware`.
- Allowed origins: matching regex `http://localhost:\d+`.
- Allowed methods: `POST`, `GET`.
- Allowed headers: `*`.

## POST /api/users/register (planned, disabled)

- Description: Planned user registration endpoint; currently disabled in code.
- Method: `POST`
- Path: `/api/users/register`
- Request Body: `UserRegisterRequest` Pydantic model (defined in `Back_Lumiere/dev/app/main.py`) with fields:
  - `username` (string)
  - `email` (string, validated as email)
  - `password` (string)
  - `profile` (optional object) with `first_name`, `last_name`, `age`, `skin_type_self_assessed`

### Behavior (current)

- The route is declared in the code but intentionally hidden from OpenAPI (`include_in_schema=False`) and immediately returns HTTP `501 Not Implemented` with JSON content:

```json
{ "detail": "User registration feature is planned but currently disabled." }
```

### Notes (when enabling)

- Move the Pydantic models to a dedicated `schemas` or `models` module for clarity.
- Implement secure password hashing, uniqueness checks for `username`/`email`, input validation, and optional email verification.
- Add persistence (e.g., a new `users` table) and proper error handling.
- Consider exposing the endpoint in OpenAPI only when the feature is ready (remove `include_in_schema=False`).

## 7. Known Limitations (已知限制)

- No authentication or authorization implemented.
- Only local development origins are allowed by CORS.
- Analysis result endpoint uses a simple ID lookup; no paging or search is supported.
- The response object schema is not formally enforced by Pydantic on the API layer.

## 8. Data Persistence

- `save_analysis(result: dict) -> str` generates an `analysis_id` using `secrets.token_hex(4)` prefixed with `ana_`.
- It stores the result in the `analyses` table.
- `get_analysis(analysis_id: str) -> dict | None` reads from `analyses` and returns the saved JSON plus `id` and `created_at`.

## 9. API Response Structure (API 回應結構)

### 9.1 Analysis Result Root Structure

The analysis result returned by `POST /api/analyze` and `GET /api/analyze/{analyze_id}` is a JSON object representing the skin analysis report.

Common top-level fields:

- `id`: string analysis identifier (added after persistence)
- `created_at`: string [ISO 8601] (only present on GET retrieval)
- `tom_geral_fitzpatrick`: integer (1-6)
- `tom_geral_hex`: string `#RRGGBB`
- `fitzpatrick_source`: string `bisenet` or `openai`
- `subtom_predominante`: string `quente`, `frio`, or `neutro`
- `regioes`: object with per-region analysis
- `comparacao_tons`: object with tone comparisons
- `imperfeicoes`: array of detected imperfections
- `recommendations`: array of product recommendations
- `skin_tone`: object with BiSeNet full-face skin tone stats

### 9.2 Region Object Example

Each region entry may include:

- `tom_hex`: string `#RRGGBB`
- `tom_fitzpatrick`: integer (1-6)
- `subtom`: string
- `oleosidade`: string
- `imperfeicoes`: array
- `uniformidade`: integer
- `notas`: string

### 9.3 Comparison Object Example

Each comparison entry includes:

- `delta`: float
- `nivel`: `alto`, `moderado`, or `baixo`

### 9.4 Recommendation Object Example

Each recommendation entry includes:

- `brand`: string
- `shade_name`: string
- `shade_code`: string
- `undertone`: string
- `shade_hex`: string
- `price_range`: string
- `where_to_buy`: string

### 9.5 Skin Tone Object Example

Each skin tone entry includes:

- `mean_rgb`: array or null
- `mean_hex`: string or null
- `median_rgb`: array or null
- `median_hex`: string or null
- `num_skin_pixels`: integer

## 10. Determination Basis and Sources (判斷依據與來源)

| Field | Source Code | Notes |
|---|---|---|
| `GET /`, CORS | `Back_Lumiere/dev/app/main.py` | Routes and middleware are defined here. |
| `POST /api/analyze`, `GET /api/analyze/{analyze_id}` | `Back_Lumiere/dev/app/main.py` | API endpoints and request validation. |
| Result persistence | `Back_Lumiere/dev/app/data_service.py` | `save_analysis()` and `get_analysis()`. |
| Pipeline execution | `Back_Lumiere/dev/app/pipeline.py` | `run_pipeline()` executes region analysis and BiSeNet skin tone analysis. |
| Skin tone analysis | `Back_Lumiere/dev/app/skin_tone_analyzer.py` | BiSeNet segmentation and skin-tone calculation. |
| Region analysis / prompt result | `Back_Lumiere/dev/app/vision.py` | Region-by-region analysis via OpenAI. |
| Recommendation logic | `Back_Lumiere/dev/app/recommendations.py` | Matches shade recommendations by tone and undertone. |

## 11. Notes

- `POST /api/analyze` returns the full analysis object and appends `id` after saving.
- `GET /api/analyze/{analyze_id}` returns the persisted object with `created_at`.
- If the database is unreachable, health checks return `db: "unreachable"` but still return HTTP 200.

