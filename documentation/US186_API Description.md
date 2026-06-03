# API Contract

## 1. GET /

- Description: Service health check (服務健康檢查)
- Method: `GET`
- Path: `/`
- Response (200):
  ```json
  {
    "status": "ok",
    "service": "skin-analyzer"
  }
  ```

## 2. POST /api/analyze

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
  - Content: Complete analysis result (see [API 200 Response Structure](#api-200-response-structure) below) (完整分析結果)
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
  - Reason: Internal error in image analysis pipeline (影像分析流程內部錯誤)

## 3. Frontend API Usage (前端 API 用法)

- Frontend base URL: `http://localhost:8001` (default) (前端呼叫位址；預設)
- Frontend service file: `Front_Lumiere/src/services/api.js`
- Wrapper function:
  - `analyzeImage(file)`
  - Sends `multipart/form-data` to `/api/analyze`

## 4. Known Limitations (已知限制)

- Only one main analysis API (僅有單一主要分析 API)
- No authentication or authorization implemented (目前無額外認證或授權機制)
- Only accessible from localhost or matching `http://localhost:\d+` origins (目前只開放給 localhost 或符合 `http://localhost:\d+` 的來源)

## 5. Determination Basis and Sources (判斷依據與來源)

### Backend API Sources (後端 API 來源)

- Main source: `Back_Lumiere/dev/app/main.py`
- This file defines the FastAPI application `app = FastAPI(...)` (這個檔案定義了 FastAPI 應用)
- `GET /` is defined by `@app.get("/")` (`@app.get("/")` 判斷出)
- `POST /api/analyze` is defined by `@app.post("/api/analyze")` (`@app.post("/api/analyze")` 判斷出)

### Backend Determination Details (後端判斷細節)

- Request parameters for `POST /api/analyze` are inferred from `analyze(file: UploadFile = File(...))` (由參數推斷)
- Supported formats are inferred from `file.content_type not in ALLOWED`, where `ALLOWED` includes `image/jpeg`, `image/png`, `image/webp` (由格式判斷)
- Maximum file size is inferred from `if len(contents) > 10 * 1024 * 1024` = 10 MB (判斷出)
- Error responses `400`, `413`, `422`, `500` are inferred from `HTTPException(...)` and exception handling (推斷)

### Frontend Call Sources (前端呼叫來源)

- Main source: `Front_Lumiere/src/services/api.js`
- Backend path `/api/analyze` is confirmed by `api.post('/api/analyze', formData, ...)` (確認)
- Request field name `file` is inferred from `formData.append('file', file)` (推斷)
- Multipart upload is confirmed by `headers: { 'Content-Type': 'multipart/form-data' }` (推斷)

### Supplementary Document Sources (補充文件來源)

- `Back_Lumiere/README.md` also lists `GET /` and `POST /api/analyze`, consistent with the code (與程式碼定義一致)
- This document serves as a reference to confirm that API endpoints and limitations remain unchanged (確認 API 端點與限制不變)

## 6. API 200 Response Structure (API 200 回應完整結構)

### Root Level Structure (根層結構)

```json
{
  "tom_geral_fitzpatrick": "integer (1-6) [Fitzpatrick skin type]",
  "tom_geral_hex": "string ('#RRGGBB') [Overall skin tone hex]",
  "fitzpatrick_source": "string ('bisenet' | 'claude') [Data source]",
  "subtom_predominante": "string ('quente' | 'frio' | 'neutro') [Predominant undertone]",
  "regioes": "object [Region-by-region analysis]",
  "comparacao_tons": "object [Tone comparisons between regions]",
  "imperfeicoes": "array [All detected imperfections]",
  "recommendations": "array [Product shade recommendations]",
  "skin_tone": "object [BiSeNet full-face analysis]"
}
```

### Field Details (欄位詳細說明)

#### Top-Level Fields (頂層欄位)

| Field Name | Type | Description (說明) | Example |
|---------|------|------|------|
| `tom_geral_fitzpatrick` | `int` | Fitzpatrick skin type (1=light ~ 6=dark) (膚色類型) | `3` |
| `tom_geral_hex` | `str` | Overall skin hex, prioritizing BiSeNet result (整體膚色十六進制值) | `"#c68b6e"` |
| `fitzpatrick_source` | `str` | Fitzpatrick type source (來源) | `"bisenet"` or `"claude"` |
| `subtom_predominante` | `str` | Predominant skin undertone (主要膚色冷暖調) | `"neutro"` / `"quente"` / `"frio"` |

#### Regioes (Region Analysis) (區域分析)

Structure: `{ "region_name": {...}, "region_name": {...}, ... }`

**Supported region names (支援的區域名稱)**: `testa`, `bochecha_e`, `bochecha_d`, `nariz`, `queixo`, etc.

Each region structure:

```json
{
  "tom_hex": "#RRGGBB [Hex color for this region]",
  "tom_fitzpatrick": "integer (1-6) [Fitzpatrick type]",
  "subtom": "'quente' | 'frio' | 'neutro' [Undertone]",
  "oleosidade": "'seco' | 'normal' | 'misto' | 'oleoso' [Oiliness level]",
  "imperfeicoes": "[Array of imperfections detected]",
  "uniformidade": "integer (0-10) [Uniformity score]",
  "notas": "string (max 1 sentence) [Observation in English]"
}
```

Example region:

```json
{
  "tom_hex": "#d0a070",
  "tom_fitzpatrick": 3,
  "subtom": "quente (warm)",
  "oleosidade": "normal",
  "imperfeicoes": [
    {"tipo": "poro", "intensidade": "moderado"},
    {"tipo": "linha", "intensidade": "leve"}
  ],
  "uniformidade": 7,
  "notas": "Slightly warmer than average; open pores."
}
```

#### Comparacao_tons (Tone Comparison) (色調對比)

Structure: `{ "region1_vs_region2": {...}, ... }`

Each comparison structure:

```json
{
  "delta": "float (color difference, rounded to 2 decimal places) [色差值]",
  "nivel": "'alto' | 'moderado' | 'baixo' [Difference level]"
}
```

Level determination:
- `delta > 20` → `"alto"` (high)
- `10 < delta ≤ 20` → `"moderado"` (moderate)
- `delta ≤ 10` → `"baixo"` (low)

#### Imperfeicoes (All Imperfections List) (所有缺陷清單)

Structure: `[{...}, {...}, ...]`

Each imperfection structure:

```json
{
  "tipo": "'acne' | 'mancha' | 'poro' | 'linha' | 'vermelhidao' | 'outro' [Type]",
  "intensidade": "'leve' | 'moderado' | 'intenso' [Severity]",
  "regiao": "'testa' | 'bochecha_e' | ... [Region where detected]"
}
```

#### Recommendations (Product Recommendations) (產品推薦)

Structure: `[{...}, {...}, ...]`

Each recommendation structure:

```json
{
  "brand": "string [Brand name, e.g. 'Fenty Beauty']",
  "shade_name": "string [Shade name, e.g. '110N']",
  "shade_code": "string [Shade code]",
  "undertone": "'quente' | 'frio' | 'neutro' [Undertone preference]",
  "shade_hex": "string [Hex color, e.g. '#c9976f']",
  "price_range": "string [Price range, e.g. '$38–$42']",
  "where_to_buy": "string [Purchase link/URL]"
}
```

Recommendation criteria:
- Prioritizes matching skin undertone preference (優先匹配膚色冷暖調偏好)
- Sorted by color distance proximity (按色差遠近排序)
- Maximum 1 shade recommendation per brand (每個品牌最多推薦 1 個色號)

#### Skin_tone (BiSeNet Full-Face Analysis) (BiSeNet 整臉膚色分析)

Structure:

```json
{
  "mean_rgb": "[R, G, B] | null [Mean RGB of skin pixels]",
  "mean_hex": "string '#RRGGBB' | null [Mean hex value]",
  "median_rgb": "[R, G, B] | null [Median RGB of skin pixels]",
  "median_hex": "string '#RRGGBB' | null [Median hex value]",
  "num_skin_pixels": "integer [Number of detected skin pixels]"
}
```

Explanation:
- `mean_*` is the average of all detected skin pixels (所有膚色像素的平均值)
- `median_*` is the median (more representative of typical skin tone) (更能代表典型膚色)
- Values may be `null` or `0` if model cannot detect skin pixels (若無偵測到膚色像素)

### Complete Response Example (完整範例回應)

```json
{
  "tom_geral_fitzpatrick": 3,
  "tom_geral_hex": "#d4a574",
  "fitzpatrick_source": "bisenet",
  "subtom_predominante": "neutro",
  "regioes": {
    "testa": {
      "tom_hex": "#d0a070",
      "tom_fitzpatrick": 3,
      "subtom": "quente",
      "oleosidade": "normal",
      "imperfeicoes": [
        {"tipo": "poro", "intensidade": "moderado"},
        {"tipo": "linha", "intensidade": "leve"}
      ],
      "uniformidade": 7,
      "notas": "Slightly warmer than average; open pores."
    },
    "bochecha_e": {
      "tom_hex": "#d9ab7a",
      "tom_fitzpatrick": 3,
      "subtom": "neutro",
      "oleosidade": "normal",
      "imperfeicoes": [],
      "uniformidade": 8,
      "notas": "Uniform skin tone, minimal imperfections."
    },
    "nariz": {
      "tom_hex": "#c89865",
      "tom_fitzpatrick": 4,
      "subtom": "frio",
      "oleosidade": "oleoso",
      "imperfeicoes": [
        {"tipo": "poro", "intensidade": "intenso"}
      ],
      "uniformidade": 6,
      "notas": "Oily T-zone with enlarged pores."
    }
  },
  "comparacao_tons": {
    "testa_vs_bochecha_e": {
      "delta": 12.34,
      "nivel": "moderado"
    },
    "nariz_vs_testa": {
      "delta": 25.67,
      "nivel": "alto"
    }
  },
  "imperfeicoes": [
    {"tipo": "poro", "intensidade": "moderado", "regiao": "testa"},
    {"tipo": "linha", "intensidade": "leve", "regiao": "testa"},
    {"tipo": "poro", "intensidade": "intenso", "regiao": "nariz"}
  ],
  "recommendations": [
    {
      "brand": "Fenty Beauty",
      "shade_name": "330N",
      "shade_code": "330N",
      "undertone": "neutro",
      "shade_hex": "#c9976f",
      "price_range": "$38–$42",
      "where_to_buy": "https://www.fentybeauty.com"
    },
    {
      "brand": "MAC",
      "shade_name": "NC35",
      "shade_code": "NC35",
      "undertone": "quente",
      "shade_hex": "#c9966a",
      "price_range": "$35–$40",
      "where_to_buy": "https://www.maccosmetics.com"
    }
  ],
  "skin_tone": {
    "mean_rgb": [212, 165, 116],
    "mean_hex": "#d4a574",
    "median_rgb": [215, 168, 119],
    "median_hex": "#d7a877",
    "num_skin_pixels": 45000
  }
}
```

### Data Sources and Determination Basis (資料來源與判斷依據)

| Field (欄位) | Source Code (來源程式) | Determination Method (判斷方式) |
|------|---------|---------|
| `tom_geral_fitzpatrick` | [color_utils.py](color_utils.py) | Determined by `hex_to_fitzpatrick()` function based on color distance, prioritizing BiSeNet result (由...根據色距判定，優先使用 BiSeNet 結果) |
| Region `tom_hex` etc. (各區域...) | [vision.py](vision.py) | Analyzed by Claude API on region crops (Claude API 根據區域裁圖分析) |
| `regioes` structure | [PROMPT](vision.py#L11) | JSON structure returned by Claude (defined in prompt) (Claude 傳回的 JSON 結構；定義在提示詞中) |
| `comparacao_tons` | [color_utils.py](color_utils.py) | Calculated by `color_delta()` function; `nivel` determined by threshold rules (計算色差；依規則判定) |
| `imperfeicoes` | [vision.py](vision.py) | Aggregated from region Claude analysis + region field added (各區域 Claude 分析結果統合；加入欄位) |
| `recommendations` | [recommendations.py](recommendations.py) | `get_recommendations()` matches by Fitzpatrick/color/undertone (依...匹配) |
| `skin_tone` | [skin_tone_analyzer.py](skin_tone_analyzer.py) | BiSeNet model segments skin pixels, calculates mean/median RGB (模型分割膚色像素，計算) |

