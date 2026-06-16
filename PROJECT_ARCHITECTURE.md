# Lumiere Project Architecture & Tech Stack

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER (Web Browser)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS/HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FRONT_LUMIERE (Port 5173)                   │
│                    React + Vite + Vite Router                   │
├─────────────────────────────────────────────────────────────────┤
│  Components Layer                    State Management Layer      │
│  ├─ UploadZone                      ├─ useAnalysis Hook         │
│  ├─ CameraCapture                   ├─ useCamera Hook           │
│  ├─ FacePreview                     └─ LanguageContext          │
│  ├─ AnalysisResult                                              │
│  ├─ Recommendations                                             │
│  ├─ RegionCard                                                  │
│  └─ ErrorBoundary                                               │
├─────────────────────────────────────────────────────────────────┤
│  Services Layer                                                  │
│  └─ api.js (Axios HTTP Client)                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ POST /api/analyze
                             │ GET /api/analyze/{id}
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  BACK_LUMIERE (Port 8001)                       │
│                     FastAPI + Uvicorn                           │
├─────────────────────────────────────────────────────────────────┤
│  API Routes (main.py)                                           │
│  ├─ GET /                    (Health Check)                     │
│  ├─ POST /api/analyze        (Image Analysis)                   │
│  └─ GET /api/analyze/{id}    (Retrieve Results)                 │
├─────────────────────────────────────────────────────────────────┤
│  Orchestration Layer (pipeline.py)                              │
│  ├─ Preprocess Image                                            │
│  ├─ Run MediaPipe + BiSeNet in parallel                         │
│  └─ Aggregate Results                                           │
├─────────────────────────────────────────────────────────────────┤
│  Processing Layer                                               │
│  ├─ mediapipe_utils.py        ├─ skin_tone_analyzer.py          │
│  │   ├─ get_landmarks()       │   ├─ BiSeNet Model             │
│  │   └─ extract_region_crops()│   ├─ RGB Extraction            │
│  │                             │   └─ Fitzpatrick Analysis      │
│  ├─ vision.py                 ├─ recommendations.py             │
│  │   └─ Claude Vision API     │   └─ Foundation Matching        │
│  │                             │                                 │
│  └─ image_utils.py, color_utils.py                              │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                      │
│  └─ data_service.py (SQLite ORM)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
    ┌────────┐          ┌──────────┐       ┌───────────────┐
    │ SQLite │          │  Claude  │       │ Model Weights │
    │   DB   │          │  Vision  │       │   (BiSeNet)   │
    │ (/.db) │          │   API    │       │  (.pth files) │
    └────────┘          └──────────┘       └───────────────┘
```

---

## 📊 Data Flow Diagram

### User Analysis Workflow
```
┌─────────────┐
│   Upload    │
│ Image/Photo │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Image Validation   │
│  & Preprocessing    │
└──────┬──────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│       Face Detection (MediaPipe)             │
│  Detect landmarks & segment regions          │
└──────┬───────────────────────────────────────┘
       │
       ├─────────────────────────────┬──────────────────┐
       │                             │                  │
       ▼                             ▼                  ▼
┌─────────────┐            ┌──────────────┐    ┌──────────────┐
│  Forehead   │            │  Cheeks      │    │  Nose/Chin   │
│  Crop       │            │  Crops       │    │  Crops       │
└─────────────┘            └──────────────┘    └──────────────┘
       │                             │                  │
       └─────────────┬───────────────┴──────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    ┌──────────────┐      ┌──────────────────────┐
    │ Claude Vision│      │ BiSeNet Skin Tone    │
    │ Analysis     │      │ & RGB Extraction     │
    │ (per-region) │      │ (full image)         │
    └──────┬───────┘      └──────┬───────────────┘
           │                     │
           └─────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Aggregate Results   │
          │  - Region findings   │
          │  - Skin tone data    │
          │  - Color values      │
          └──────┬───────────────┘
                 │
                 ▼
          ┌──────────────────────┐
          │ Generate Final       │
          │ Report + Recs        │
          └──────┬───────────────┘
                 │
                 ▼
          ┌──────────────────────┐
          │ Save to Database     │
          │ Generate ID          │
          └──────┬───────────────┘
                 │
                 ▼
          ┌──────────────────────┐
          │ Return to Frontend   │
          │ Display Results      │
          └──────────────────────┘
```

---

## 🛠️ Technology Stack

### Backend

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI 0.111 | REST API, async support, auto OpenAPI docs |
| **Server** | Uvicorn | ASGI server for FastAPI |
| **ML/CV** | MediaPipe 0.10.33 | Face detection, landmark extraction |
| **Segmentation** | PyTorch 2.0+, BiSeNet | Skin extraction, tone analysis |
| **Image Processing** | OpenCV, Pillow | Image validation, preprocessing |
| **AI Integration** | Anthropic SDK 0.40 | Claude Vision API calls |
| **Database** | SQLAlchemy, SQLite | Data persistence |
| **Validation** | Pydantic 2.12 | Data validation, settings |
| **Utilities** | NumPy, SciPy | Numerical computations |

### Frontend

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | React 18+ | UI components, reactivity |
| **Build Tool** | Vite | Fast dev server, optimized builds |
| **HTTP Client** | Axios | REST API calls |
| **State Mgmt** | React Hooks | Local state management |
| **i18n** | Custom Context | Multi-language support |
| **Styling** | CSS3 | Component styles |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Container** | Docker (optional) | Deployment packaging |
| **Package Mgmt** | pip, npm | Dependency management |
| **Version Control** | Git | Code versioning |

---

## 📦 Key Dependencies

### Backend Dependencies
```
fastapi==0.111.0              # Web framework
torch>=2.0.0                  # Deep learning
torchvision>=0.15.0           # CV models
mediapipe==0.10.33            # Face detection
opencv-python-headless        # Image processing
pillow==10.4.0                # Image library
anthropic==0.40.0             # Claude API
sqlalchemy                    # ORM
pydantic==2.12.5              # Data validation
python-dotenv                 # .env support
```

### Frontend Dependencies
```
react@18+                     # UI library
vite                          # Build tool
axios                         # HTTP client
react-dom                     # React rendering
```

---

## 🔄 Component Relationships

```
┌──────────────────────────────────────────────────────────────┐
│                     PIPELINE (Main Orchestrator)              │
├──────────────────────────────────────────────────────────────┤
│ run_pipeline(img_rgb) → Coordinates entire analysis flow     │
│                                                               │
│  ├─ image_utils.preprocess()                                 │
│  │   └─ Validate & resize image                              │
│  │                                                            │
│  ├─ mediapipe_utils.get_landmarks()                          │
│  │   └─ Detect 468 face landmarks                            │
│  │                                                            │
│  ├─ mediapipe_utils.extract_region_crops()                   │
│  │   └─ Extract 4 regions + create Base64                    │
│  │                                                            │
│  ├─ PARALLEL:                                                │
│  │   ├─ vision.analyze_region() × 4 regions                  │
│  │   │   ├─ Call Claude Vision API                           │
│  │   │   └─ Parse pores, acne, redness, etc.                 │
│  │   │                                                        │
│  │   └─ skin_tone_analyzer.analyze_skin_tone()               │
│  │       ├─ Load BiSeNet model                               │
│  │       ├─ Extract skin mask                                │
│  │       ├─ Calculate RGB + HSV                              │
│  │       └─ Fitzpatrick classification                       │
│  │                                                            │
│  └─ color_utils.build_final_report()                         │
│      └─ Aggregate + format results                           │
│                                                               │
│  └─ data_service.save_analysis()                             │
│      └─ Persist to DB, return ID                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔗 API Contract Simplified

### Input/Output Examples

**Request:**
```bash
curl -X POST -F "file=@face.jpg" http://localhost:8001/api/analyze
```

**Response (simplified):**
```json
{
  "id": "ana_a1b2c3d4",
  "skin_tone": {
    "fitzpatrick_scale": "Type III-IV",
    "undertone": "warm",
    "hex_value": "#C08552",
    "rgb": [192, 133, 82]
  },
  "regions": {
    "forehead": {
      "pores": "moderate",
      "acne": "none",
      "redness": "mild"
    },
    "left_cheek": {...},
    "right_cheek": {...},
    "nose": {...},
    "chin": {...}
  },
  "recommendations": {
    "foundation": [
      {"brand": "Fenty Beauty", "shade": "140W", "reason": "..."}
    ]
  }
}
```

---

## 🚀 Deployment Architecture

### Development
```
localhost:5173 (Frontend) ←→ localhost:8001 (Backend)
```

### Production
```
CDN/Static Hosting     ← Frontend (React build)
  └─ HTTPS ─→ API Server (Uvicorn + Reverse Proxy)
              └─ SQLite DB (or PostgreSQL)
```

### Docker Approach
```dockerfile
# Backend Dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY Back_Lumiere/dev/app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

# Frontend Dockerfile (multi-stage)
FROM node:18 AS build
WORKDIR /app
COPY package.json .
RUN npm install
COPY Front_Lumiere/src ./src
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
```

---

## 📈 Performance Considerations

### Bottlenecks & Optimizations

| Bottleneck | Current | Optimization |
|-----------|---------|--------------|
| Claude Vision API latency | ~2-3s per region | Batch regions, async concurrent calls |
| BiSeNet inference | ~1-2s | GPU acceleration (CUDA) |
| Image preprocessing | ~200-500ms | Edge computing, browser-side crop |
| Network | Varies | Compression, caching |

### Recommended Improvements
1. **Add request caching** - Cache duplicate analysis results
2. **Batch processing** - Group multiple region requests
3. **GPU support** - Enable CUDA for BiSeNet
4. **Image compression** - Browser-side compression before upload
5. **CDN** - Serve frontend assets from CDN

---

## 🔐 Security Considerations

### Current Implementation
- ✅ File type validation (JPEG, PNG, WebP only)
- ✅ File size limit (10MB max)
- ✅ CORS enabled for localhost development
- ✅ Error handling without exposing internal details

### Security Improvements Needed
- ❌ Authentication & Authorization
- ❌ Rate limiting
- ❌ Input sanitization
- ❌ HTTPS enforcement (production)
- ❌ API key rotation

### Recommended Security Enhancements
```python
# Example: Add rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/analyze")
@limiter.limit("5/minute")  # 5 requests per minute per IP
async def analyze(file: UploadFile):
    ...
```

---

## 📊 Monitoring & Logging

### Recommended Setup
```python
# Logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics to Track
- API response times
- Claude Vision API costs
- Database query performance
- Error rates and types
- User session analytics

---

## 🎯 Development Roadmap

### Phase 1: MVP ✅
- [x] Image upload & camera capture
- [x] Face detection & region extraction
- [x] Per-region AI analysis
- [x] Skin tone detection
- [x] Foundation recommendations
- [x] Multi-language UI

### Phase 2: Enhancement
- [ ] User authentication & profiles
- [ ] Analysis history
- [ ] More makeup brands
- [ ] Shade customization
- [ ] Advanced filters (lighting correction)

### Phase 3: ML Improvements
- [ ] Fine-tune BiSeNet on diverse datasets
- [ ] Add disease classification (melasma, vitiligo, port wine stain)
- [ ] Local inference (privacy-first)
- [ ] Model quantization for edge devices

### Phase 4: Enterprise
- [ ] Mobile apps (React Native)
- [ ] API for third parties
- [ ] Salon/clinic management dashboard
- [ ] Analytics & reporting

---

**For detailed implementation, refer to:**
- [QUICKSTART_ZH.md](./QUICKSTART_ZH.md) - Quick start guide (Chinese)
- [Back_Lumiere/documentation/US186_API Description.md](./Back_Lumiere/documentation/US186_API Description.md) - Full API spec
- [Back_Lumiere/documentation/US160_SegFormer.md](./Back_Lumiere/documentation/US160_SegFormer.md) - Model details
