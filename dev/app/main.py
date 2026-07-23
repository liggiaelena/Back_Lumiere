import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.logging_config import setup_logging
from app.pipeline import run_pipeline
from app.image_utils import load_and_validate
from app.db import engine
from app.data_service import save_analysis, get_analysis

setup_logging()
logger = logging.getLogger(__name__)

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
    logger.info("Health check requested (db=%s)", db_status)
    return {"status": "ok", "service": "skin-analyzer", "db": db_status}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), lang: str = "en"):
    ALLOWED = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in ALLOWED:
        raise HTTPException(400, detail="Invalid format. Use JPG, PNG or WebP.")

    contents = await file.read()
    logger.info("Received image for analysis (%d bytes, lang=%s)", len(contents), lang)

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, detail="Image too large. Maximum 10MB.")

    try:
        img_rgb = load_and_validate(contents)
        result = await run_pipeline(img_rgb, lang=lang)
        analysis_id = save_analysis(result)
        result["id"] = analysis_id
        logger.info("Analysis completed successfully (id=%s)", analysis_id)
        return JSONResponse(content=result)
    except ValueError as e:
        logger.error("Analysis rejected: %s", e)
        raise HTTPException(422, detail=str(e))
    except Exception:
        logger.error("Internal error while analyzing the image", exc_info=True)
        raise HTTPException(500, detail="Internal error while analyzing the image.")

@app.get("/api/analyze/{analyze_id}")
def get_analyze(analyze_id: str):
    try:
        result = get_analysis(analyze_id)
    except Exception:
        logger.error("Internal error while fetching analysis %s", analyze_id, exc_info=True)
        raise HTTPException(500, detail="Internal error while fetching the analysis.")
    if result is None:
        raise HTTPException(404, detail="Analyze result not found.")
    logger.info("Fetched analysis result (id=%s)", analyze_id)
    return JSONResponse(content=result)
