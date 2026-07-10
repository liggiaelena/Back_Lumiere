from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.pipeline import run_pipeline
from app.image_utils import load_and_validate
from app.db import engine
from app.data_service import save_analysis, get_analysis
import traceback

app = FastAPI(title="Skin Analyzer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],            # Temporarily change it to "*" to ensure the frontend can connect. During production deployment, change it to the actual frontend domain.
    allow_credentials=True,         
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], 
    allow_headers=["*"],
)


# --- User registration (planned, disabled) ---
class _Profile(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    skin_type_self_assessed: Optional[str] = None


class UserRegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    profile: Optional[_Profile] = None


@app.post(
    "/api/users/register",
    include_in_schema=False,
    summary="(disabled) Register a new user",
)
async def register_user(req: UserRegisterRequest):
    """User registration endpoint (disabled).

    This route is intentionally disabled and returns HTTP 501. When enabled,
    it should validate input, create a user record, and return the created
    user's ID or a suitable response. For now it returns a clear 501 JSON.
    """
    return JSONResponse(
        content={"detail": "User registration feature is planned but currently disabled."},
        status_code=501,
    )


@app.get("/")
def health():
    db_status = "ok"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_status = "unreachable"
    return {"status": "ok", "service": "skin-analyzer", "db": db_status}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), lang: str = "en"):
    ALLOWED = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in ALLOWED:
        raise HTTPException(400, detail="Invalid format. Use JPG, PNG or WebP.")

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, detail="Image too large. Maximum 10MB.")

    try:
        img_rgb = load_and_validate(contents)
        result = await run_pipeline(img_rgb, lang=lang)
        analysis_id = save_analysis(result)
        result["id"] = analysis_id
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, detail="Internal error while analyzing the image.")

@app.get("/api/analyze/{analyze_id}")
def get_analyze(analyze_id: str):
    try:
        result = get_analysis(analyze_id)
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, detail="Internal error while fetching the analysis.")
    if result is None:
        raise HTTPException(404, detail="Analyze result not found.")
    return JSONResponse(content=result)
