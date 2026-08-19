import logging
import asyncio
import secrets
from datetime import datetime
from pathlib import Path
from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.logging_config import setup_logging
from app.pipeline import run_pipeline
from app.image_utils import load_and_validate
from app.db import engine
from app.data_service import (
    AnalysisQueueFullError,
    activate_analysis_job,
    create_analysis_job,
    create_or_get_recommendation_job,
    ensure_analysis_job_schema,
    ensure_analysis_ownership,
    get_analysis,
    get_analysis_status,
    get_recommendation_job,
    list_user_analyses,
    mark_analysis_failed,
    recommendation_response,
)
from app.auth import create_access_token, decode_access_token, verify_password
from app.user_service import (
    create_user,
    ensure_users_table,
    find_user_by_email,
    find_user_by_id,
)
from app.gpt_recommendations import (
    DEFAULT_ALLERGEN_OPTIONS,
)
from app.fallback_catalog.product_service import ensure_product_tables
from app.job_workers import start_workers, stop_workers

setup_logging()
logger = logging.getLogger(__name__)

from app.config import log_startup_config, settings
log_startup_config()

app = FastAPI(title="Skin Analyzer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],            # Temporarily change it to "*" to ensure the frontend can connect. During production deployment, change it to the actual frontend domain.
    allow_credentials=True,         
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], 
    allow_headers=["*"],
)


class _Profile(BaseModel):
    first_name: Optional[str] = Field(default=None, max_length=100)
    last_name: Optional[str] = Field(default=None, max_length=100)
    age: Optional[int] = Field(default=None, ge=13, le=120)
    skin_type_self_assessed: Optional[str] = Field(default=None, max_length=50)


class UserRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    profile: Optional[_Profile] = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, password: str) -> str:
        if not any(char.isalpha() for char in password) or not any(
            char.isdigit() for char in password
        ):
            raise ValueError("Password must include at least one letter and one number.")
        return password


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    skin_type_self_assessed: Optional[str] = None
    created_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RecommendationRequest(BaseModel):
    excluded_allergens: list[str] = Field(default_factory=list)
    lang: str = Field(default="en", max_length=10)
    force_fallback: bool = False


bearer_scheme = HTTPBearer(auto_error=False)


def public_user(user: dict) -> dict:
    return {key: value for key, value in user.items() if key != "password_hash"}


def current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = find_user_by_id(decode_access_token(credentials.credentials))
    if not user:
        raise HTTPException(status_code=401, detail="User account no longer exists.")
    return user


def optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[dict]:
    if not credentials:
        return None
    return current_user(credentials)


@app.on_event("startup")
async def initialize_database():
    ensure_users_table()
    ensure_analysis_ownership()
    ensure_analysis_job_schema()
    ensure_product_tables()
    await start_workers()


@app.on_event("shutdown")
async def shutdown_workers():
    await stop_workers()


@app.post(
    "/api/users/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
@app.post(
    "/api/auth/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(req: UserRegisterRequest):
    try:
        user = create_user(req)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    token, expires_in = create_access_token(user["id"])
    return {
        "access_token": token,
        "expires_in": expires_in,
        "user": public_user(user),
    }


@app.post("/api/auth/login", response_model=AuthResponse)
async def login_user(req: UserLoginRequest):
    user = find_user_by_email(str(req.email))
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    token, expires_in = create_access_token(user["id"])
    return {
        "access_token": token,
        "expires_in": expires_in,
        "user": public_user(user),
    }


@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user(user: dict = Depends(current_user)):
    return user


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

def _parse_allergens(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = __import__("json").loads(raw)
        if isinstance(value, list):
            return sorted({str(item).strip() for item in value if str(item).strip()})
    except (ValueError, TypeError):
        pass
    return sorted({item.strip() for item in raw.split(",") if item.strip()})


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    lang: str = "en",
    user: Optional[dict] = Depends(optional_current_user),
    excluded_allergens: str = Form(default=""),
):
    ALLOWED = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in ALLOWED:
        raise HTTPException(400, detail="Invalid format. Use JPG, PNG or WebP.")

    contents = await file.read()
    logger.info("Received image for analysis (%d bytes, lang=%s)", len(contents), lang)

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(413, detail=f"Image too large. Maximum {settings.max_image_size_mb}MB.")

    try:
        # Reject invalid images before consuming one of the ten queue slots.
        await asyncio.to_thread(load_and_validate, contents)
    except ValueError as e:
        logger.error("Analysis rejected: %s", e)
        raise HTTPException(422, detail=str(e))

    upload_dir = Path(settings.upload_storage_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[file.content_type]
    image_path = upload_dir / f"upload_{secrets.token_hex(16)}{suffix}"
    try:
        job = await asyncio.to_thread(
            create_analysis_job,
            user_id=user["id"] if user else None,
            image_path=str(image_path),
            content_type=file.content_type,
            size_bytes=len(contents),
            lang=lang,
            excluded_allergens=_parse_allergens(excluded_allergens),
            max_unfinished=settings.max_unfinished_analyses,
        )
    except AnalysisQueueFullError:
        raise HTTPException(
            429,
            detail={"code": "analysis_queue_full", "message": "Analysis queue is full. Please try again shortly.",
                    "capacity": settings.max_unfinished_analyses, "retry_after": 20},
            headers={"Retry-After": "20"},
        )
    try:
        await asyncio.to_thread(image_path.write_bytes, contents)
    except Exception as exc:
        await asyncio.to_thread(mark_analysis_failed, job["id"], f"Could not persist uploaded image: {exc}")
        raise HTTPException(500, detail="Could not save the uploaded image.")

    await asyncio.to_thread(activate_analysis_job, job["id"])
    queued_status = await asyncio.to_thread(get_analysis_status, job["id"], user["id"] if user else None)
    if queued_status:
        job["queue_position"] = queued_status.get("queue_position")

    logger.info("Analysis queued (id=%s position=%s)", job["id"], job["queue_position"])
    return JSONResponse(status_code=202, content={**job, "poll_after_seconds": 2})

@app.get("/api/analyze")
def get_history(user: dict = Depends(current_user)):
    return {"items": list_user_analyses(user["id"])}


@app.get("/api/analyze/{analyze_id}")
def get_analyze(
    analyze_id: str,
    user: Optional[dict] = Depends(optional_current_user),
):
    try:
        result = get_analysis(analyze_id, user["id"] if user else None)
    except Exception:
        logger.error("Internal error while fetching analysis %s", analyze_id, exc_info=True)
        raise HTTPException(500, detail="Internal error while fetching the analysis.")
    if result is None:
        raise HTTPException(404, detail="Analyze result not found.")
    logger.info("Fetched analysis result (id=%s)", analyze_id)
    return JSONResponse(content=result)


@app.get("/api/analyze/{analyze_id}/status")
def analyze_status(
    analyze_id: str,
    user: Optional[dict] = Depends(optional_current_user),
):
    result = get_analysis_status(analyze_id, user["id"] if user else None)
    if result is None:
        raise HTTPException(404, detail="Analyze result not found.")
    return result


@app.post("/api/analyze/{analyze_id}/recommendations", status_code=202)
async def request_recommendations(
    analyze_id: str,
    request: RecommendationRequest,
    user: Optional[dict] = Depends(optional_current_user),
):
    result = get_analysis(analyze_id, user["id"] if user else None)
    if result is None:
        raise HTTPException(404, detail="Analyze result not found.")
    if result.get("analysis_status") != "ready":
        raise HTTPException(409, detail="Analysis is not ready yet.")
    job = await asyncio.to_thread(
        create_or_get_recommendation_job,
        analyze_id,
        request.excluded_allergens,
        request.lang or result.get("lang", "en"),
        request.force_fallback,
    )
    return recommendation_response(job)


@app.get("/api/analyze/{analyze_id}/recommendations")
@app.get("/api/analyze/{analyze_id}/recommendation")
def read_recommendations(
    analyze_id: str,
    user: Optional[dict] = Depends(optional_current_user),
):
    result = get_analysis(analyze_id, user["id"] if user else None)
    if result is None:
        raise HTTPException(404, detail="Analyze result not found.")
    return {key: value for key, value in result.items() if key.startswith("recommendation")}


@app.get("/api/recommendation-jobs/{job_id}/status")
def recommendation_status(job_id: str):
    job = get_recommendation_job(job_id)
    if job is None:
        raise HTTPException(404, detail="Recommendation job not found.")
    return {"job_id": job_id, "status": job["status"], "error": job.get("last_error"),
            "poll_after_seconds": 5 if job["status"] in {"queued", "processing"} else None}


@app.get("/api/products/allergens")
def get_catalog_allergens():
    """Values the frontend can render as allergen-exclusion checkboxes."""
    return {"allergens": DEFAULT_ALLERGEN_OPTIONS}
