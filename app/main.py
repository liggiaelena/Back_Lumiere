from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.pipeline import run_pipeline
from app.image_utils import load_and_validate
import traceback

app = FastAPI(title="Skin Analyzer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "service": "skin-analyzer"}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    ALLOWED = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in ALLOWED:
        raise HTTPException(400, detail="Invalid format. Use JPG, PNG or WebP.")

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, detail="Image too large. Maximum 10MB.")

    try:
        img_rgb = load_and_validate(contents)
        result = await run_pipeline(img_rgb)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, detail="Internal error while analyzing the image.")
