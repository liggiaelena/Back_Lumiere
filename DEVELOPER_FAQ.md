# Lumiere Developer Quick Reference & FAQ

## 🚀 Quick Command Reference

### Setup (First Time Only)
```bash
# Backend setup
cd Back_Lumiere
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Create .env with ANTHROPIC_API_KEY
python dev/run.py

# Frontend setup
cd Front_Lumiere
npm install
npm run dev
```

### Daily Commands
```bash
# Terminal 1: Backend
cd Back_Lumiere
venv\Scripts\Activate.ps1
python dev/run.py              # Start API (http://localhost:8001)

# Terminal 2: Frontend
cd Front_Lumiere
npm run dev                    # Start dev server (http://localhost:5173)
```

### Testing
```bash
# Test API health
curl http://localhost:8001/

# Test with image
curl -X POST -F "file=@test.jpg" http://localhost:8001/api/analyze

# View Swagger docs
# Open http://localhost:8001/docs
```

### Build for Production
```bash
# Frontend
cd Front_Lumiere
npm run build                  # Creates optimized dist/

# Backend (using Gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8001 app.main:app
```

---

## 📁 Key File Locations

| What | Where | What's Inside |
|------|-------|---------------|
| **Main API** | `Back_Lumiere/dev/app/main.py` | Routes (GET /, POST /api/analyze, etc.) |
| **Analysis Pipeline** | `Back_Lumiere/dev/app/pipeline.py` | Orchestrates the whole analysis flow |
| **Face Detection** | `Back_Lumiere/dev/app/mediapipe_utils.py` | Landmark detection, region extraction |
| **AI Analysis** | `Back_Lumiere/dev/app/vision.py` | Claude Vision API integration |
| **Skin Tone** | `Back_Lumiere/dev/app/skin_tone_analyzer.py` | BiSeNet model + RGB extraction |
| **Recommendations** | `Back_Lumiere/dev/app/recommendations.py` | Foundation shade matching |
| **Data Storage** | `Back_Lumiere/dev/app/data_service.py` | Save/retrieve analysis results |
| **Frontend App** | `Front_Lumiere/src/App.jsx` | Main React component |
| **API Client** | `Front_Lumiere/src/services/api.js` | Axios wrapper for backend calls |
| **Analysis Hook** | `Front_Lumiere/src/hooks/useAnalysis.js` | State management for analysis |
| **Translations** | `Front_Lumiere/src/i18n/translations.js` | All UI strings (6 languages) |
| **Env Config** | `Back_Lumiere/.env` | API keys, settings (create this!) |
| **DB Schema** | `Back_Lumiere/db/init.sql` | SQLite schema |

---

## ❓ Frequently Asked Questions

### Setup Issues

#### Q: `ModuleNotFoundError: No module named 'app'`
**Cause**: Working in wrong directory or venv not activated

**Solution**:
```bash
# Make sure you're in the project root or Back_Lumiere/
cd Back_Lumiere
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate   # Linux/Mac
python dev/run.py
```

#### Q: `FileNotFoundError: .env file not found`
**Cause**: Missing API key configuration

**Solution**: Create `Back_Lumiere/.env`
```env
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
```

Get key at: https://console.anthropic.com/

#### Q: Port 8001 already in use
**Cause**: Another service using port 8001

**Solution**:
```bash
# Find what's using port 8001
netstat -ano | findstr :8001

# Kill the process (Windows)
taskkill /PID <PID> /F

# Or use different port
python -m uvicorn app.main:app --port 8002
```

#### Q: `npm: command not found`
**Cause**: Node.js not installed or not in PATH

**Solution**:
1. Download from https://nodejs.org/ (v18+)
2. Install and restart terminal
3. Verify: `node --version`

---

### Runtime Issues

#### Q: Frontend can't connect to backend (API error)
**Cause**: Backend not running or wrong URL

**Checklist**:
- [ ] Is backend running? Check terminal for `Uvicorn running on http://localhost:8001`
- [ ] Can you access `http://localhost:8001/` in browser?
- [ ] Is `VITE_API_URL` in `Front_Lumiere/.env` set correctly?
- [ ] Check browser console (F12) for exact error

**Solution**:
```bash
# Make sure backend is running
cd Back_Lumiere
python dev/run.py

# If still stuck, try different backend port
python -m uvicorn app.main:app --port 8002
# Then update Front_Lumiere/.env: VITE_API_URL=http://localhost:8002
```

#### Q: `Invalid format. Use JPG, PNG or WebP`
**Cause**: Wrong image format or corrupted file

**Solution**:
- Use JPG, PNG, or WebP only
- File size under 10MB
- Test with a valid image

#### Q: Claude API returns "401 Unauthorized"
**Cause**: Invalid or missing API key

**Solution**:
1. Check `Back_Lumiere/.env` has correct key
2. Verify key has API access at https://console.anthropic.com/
3. Restart backend after updating .env

#### Q: Image analysis takes too long (>30 seconds)
**Possible Causes**:
- Slow internet connection (Claude API latency)
- Large image (>5MB) - preprocessing takes time
- Server overloaded

**Solutions**:
1. Use smaller image (<5MB)
2. Check internet speed
3. Try simpler photo (better lighting, no background clutter)

---

### Development Questions

#### Q: How do I add a new recommendation category?
**Example**: Adding "eyeshadow" recommendations

**Steps**:
1. Update `Back_Lumiere/dev/app/recommendations.py` with eyeshadow brands/shades
2. Modify API response schema to include `eyeshadow` field
3. Update `Front_Lumiere/src/components/Recommendations/` to display new category
4. Add translations in `Front_Lumiere/src/i18n/translations.js`
5. Test with `npm run dev` + image upload

#### Q: How do I change what Claude Vision analyzes?
**Locate**: `Back_Lumiere/dev/app/vision.py`

**Current analysis**: pores, acne, redness, sun_spots, shine

**To add new attribute**:
```python
# In analyze_region function, update the prompt:
prompt = f"""
Analyze the {region} skin region for:
- Pores (none, mild, moderate, severe)
- Acne (none, few, moderate, severe)
- Redness (none, mild, moderate, severe)
- Sun spots (none, few, moderate, many)
- Shine (none, mild, moderate, severe)
- NEW ATTRIBUTE: Texture (smooth, bumpy, rough)  # Add this
"""
```

#### Q: How do I add a new language?
**Steps**:
1. Add language code to `Front_Lumiere/src/App.jsx` LANGUAGES array
2. Add translations in `Front_Lumiere/src/i18n/translations.js`
3. Test language switching in UI

**Example** (adding Japanese):
```javascript
// App.jsx
const LANGUAGES = [
  { code: 'ja', label: '日本語' },  // Add this
  ...
]

// translations.js
export const translations = {
  ja: {
    upload: { title: "画像をアップロード" },
    ...
  }
}
```

#### Q: How do I debug the analysis pipeline?
**Add logging**:
```python
# Back_Lumiere/dev/app/pipeline.py
import logging
logger = logging.getLogger(__name__)

async def run_pipeline(img_rgb) -> dict:
    logger.info("Starting pipeline...")
    img_data = preprocess(img_rgb)
    logger.info(f"Image preprocessed: {img_data['shape']}")
    
    coords = get_landmarks(img_data["array"])
    logger.info(f"Found {len(coords)} landmarks")
    
    # ...
```

Run with debug logs:
```bash
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python dev/run.py
```

#### Q: How do I test the API manually?
**Using Swagger UI** (built-in):
1. Start backend: `python dev/run.py`
2. Open http://localhost:8001/docs
3. Click "Try it out" on `/api/analyze`
4. Upload an image and see response

**Using curl**:
```bash
curl -X POST \
  -F "file=@/path/to/face.jpg" \
  http://localhost:8001/api/analyze | jq
```

**Using Postman**:
1. Create POST request to `http://localhost:8001/api/analyze`
2. Set Content-Type: multipart/form-data
3. Add file parameter with image

---

### Performance Questions

#### Q: How to make analysis faster?
**Current bottlenecks**:
1. Claude Vision API (~2-3 seconds) - can't optimize much
2. BiSeNet inference (~1-2 seconds)
3. Network latency

**Optimizations**:
```python
# 1. Enable GPU for BiSeNet
# In skin_tone_analyzer.py:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. Reduce image size for preprocessing
MAX_SIDE = 512  # Default 1024, reduce for speed
```

**Frontend caching**:
```javascript
// In useAnalysis.js, add result caching:
const cache = new Map();
const hash = await hashImage(file);  // Hash file
if (cache.has(hash)) return cache.get(hash);
const result = await api.analyzeImage(file);
cache.set(hash, result);
return result;
```

#### Q: How to handle high traffic?
**Current**: Single-threaded Uvicorn

**Solutions**:
```bash
# 1. Use Gunicorn with multiple workers
gunicorn -w 4 -b 0.0.0.0:8001 app.main:app

# 2. Add request queuing
# Or use async task queue (Celery + Redis)

# 3. Enable response caching with Redis
# Cache identical image uploads
```

---

### Database Questions

#### Q: Where are analysis results stored?
**Default**: SQLite database (auto-created)

**Location**: Check `Back_Lumiere/dev/app/db.py` for database file path

**Schema**: `Back_Lumiere/db/init.sql`

#### Q: How do I view saved analyses?
```bash
# Using sqlite3 CLI
sqlite3 analyses.db
sqlite> SELECT * FROM analyses LIMIT 5;

# Or check via API
curl http://localhost:8001/api/analyze/ana_a1b2c3d4
```

#### Q: How do I clear all analyses?
```bash
sqlite3 analyses.db
sqlite> DELETE FROM analyses;
sqlite> .quit
```

---

### Deployment Questions

#### Q: How do I deploy to production?
**Simple (Heroku/Railway)**:
```bash
# 1. Create Procfile
echo "web: uvicorn app.main:app --host 0.0.0.0 --port $PORT" > Procfile

# 2. Push to platform (follows their deployment docs)
```

**Docker (Recommended)**:
```bash
# Build
docker build -t lumiere-backend .

# Run
docker run -p 8001:8001 -e ANTHROPIC_API_KEY=xxx lumiere-backend
```

**Server (VPS)**:
```bash
# SSH into server
ssh user@server.com

# Clone repo, install deps
git clone <repo>
cd Lumiere/Back_Lumiere
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Use systemd/supervisor to keep running
# Create /etc/systemd/system/lumiere.service
```

#### Q: How do I set up CORS for production?
**Current** (localhost only):
```python
allow_origin_regex=r"http://localhost:\d+"
```

**Production**:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

## 🔍 Debugging Checklist

### Image Upload Not Working
- [ ] Browser has camera/file permission
- [ ] File is valid image (JPG/PNG/WebP)
- [ ] File size < 10MB
- [ ] Backend is running and accessible
- [ ] Check browser console (F12) for errors

### Analysis Stuck at "Analyzing..."
- [ ] Backend console shows any errors?
- [ ] API key is valid and has quota?
- [ ] Internet connection is stable?
- [ ] Check response time in browser DevTools (Network tab)

### Wrong/Mismatched Results
- [ ] Image is clear face photo?
- [ ] Good lighting (not backlit)?
- [ ] No sunglasses or hats?
- [ ] Try different image for comparison

### Frontend Not Updating Results
- [ ] Check `Network` tab in DevTools - is API returning 200?
- [ ] Are you looking at latest code? (Hard refresh: Ctrl+Shift+R)
- [ ] Check browser console for JavaScript errors
- [ ] Try `npm run dev` again (rebuild needed?)

---

## 📚 Useful Resources

### API Documentation
- **Interactive Docs**: http://localhost:8001/docs (when running)
- **Full API Spec**: [Back_Lumiere/documentation/US186_API Description.md](Back_Lumiere/documentation/US186_API Description.md)

### Official Libraries
- [MediaPipe Docs](https://developers.google.com/mediapipe)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)

### Project Docs
- [QUICKSTART_ZH.md](./QUICKSTART_ZH.md) - Full setup guide (Chinese)
- [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md) - System architecture
- [US160_SegFormer.md](./Back_Lumiere/documentation/US160_SegFormer.md) - Model details

### Common Tasks
- **Add feature**: See [QUICKSTART_ZH.md → Development](./QUICKSTART_ZH.md#开发建议)
- **Modify prompt**: See [FAQ → Claude Vision](#q-how-do-i-change-what-claude-vision-analyzes)
- **Add brand**: Edit `recommendations.py`
- **Change language**: See [FAQ → Add Language](#q-how-do-i-add-a-new-language)

---

## 🎯 Common Development Workflows

### Workflow 1: Adding Support for New Makeup Brand
```
1. Edit recommendations.py → add brand shades
2. Test with API: curl ... | jq '.recommendations.foundation'
3. Update frontend to display new brand
4. Test in browser
5. Commit changes
```

### Workflow 2: Improving Skin Tone Accuracy
```
1. Collect test images in Back_Lumiere/test_images/
2. Run analysis and compare to expected results
3. Adjust BiSeNet preprocessing or postprocessing
4. Retrain model if needed
5. Update pipeline.py to use new checkpoint
6. Validate with frontend
```

### Workflow 3: Fixing a Bug in Analysis
```
1. Reproduce issue with specific image
2. Add logging in pipeline.py
3. Run with debug mode: python -u dev/run.py
4. Check logs to identify which step fails
5. Fix the problematic module
6. Test with same image again
7. Make sure it passes other test cases
```

---

## 💡 Pro Tips

1. **Use `/docs` endpoint** - Always available when backend running, super useful
2. **Browser DevTools** - Learn to use Network tab to debug API issues
3. **Test images locally** - Keep 5-10 diverse test images for quick testing
4. **Add print statements** - Fastest way to debug Python (besides debugger)
5. **Use environment files** - Keep `.env` out of git, use `.env.example`
6. **Restart often** - Both backend and frontend after major changes
7. **Check logs first** - 99% of issues show in console/terminal logs

---

**Still stuck? Check the full documentation or ask team members! 🚀**
