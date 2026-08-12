import logging
from datetime import datetime
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
    ensure_analysis_ownership,
    get_analysis,
    list_user_analyses,
    save_analysis,
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
from app.recommendation_service import AllRecommendationStrategiesFailed, recommend_with_fallback

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
def initialize_database():
    ensure_users_table()
    ensure_analysis_ownership()
    ensure_product_tables()


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
        img_rgb = load_and_validate(contents)
        result = await run_pipeline(
            img_rgb,
            lang=lang,
            excluded_allergens=_parse_allergens(excluded_allergens),
        )
        analysis_id = save_analysis(result, user["id"] if user else None)
        result["id"] = analysis_id
        logger.info("Analysis completed successfully (id=%s)", analysis_id)
        return JSONResponse(content=result)
    except ValueError as e:
        logger.error("Analysis rejected: %s", e)
        raise HTTPException(422, detail=str(e))
    except Exception:
        logger.error("Internal error while analyzing the image", exc_info=True)
        raise HTTPException(500, detail="Internal error while analyzing the image.")

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


@app.get("/api/analyze/{analyze_id}/recommendations")
@app.get("/api/analyze/{analyze_id}/recommendation")
async def refresh_recommendations(
    analyze_id: str,
    excluded_allergens: str = Query(default=""),
    user: Optional[dict] = Depends(optional_current_user),
):
    result = get_analysis(analyze_id, user["id"] if user else None)
    if result is None:
        raise HTTPException(404, detail="Analyze result not found.")
    if result.get("recommendations_blocked"):
        return {"recommendations": [], "recommendations_reliable": False, "recommendations_blocked": True, "recommendations_status": "blocked"}

    try:
        recommendation_result = await recommend_with_fallback(
            fitzpatrick=result["tom_geral_fitzpatrick"],
            undertone=result["subtom_predominante"],
            skin_hex=result.get("tom_geral_hex") or "#c68b6e",
            condition_map=result.get("segformer_condition_map") or result.get("condition_map"),
            excluded_allergens=_parse_allergens(excluded_allergens),
            lang=result.get("lang", "en"),
        )
    except AllRecommendationStrategiesFailed as exc:
        return {
            "recommendations": [],
            "recommendations_reliable": False,
            "recommendations_catalog_source": "openai_web_search",
            "recommendations_catalog_shades_considered": None,
            "recommendations_blocked": False,
            "recommendations_status": "unavailable",
            "recommendations_error": str(exc),
        }
    return {
        "recommendations": recommendation_result["shades"],
        "recommendations_reliable": recommendation_result["reliable"],
        "recommendations_catalog_source": recommendation_result["catalog_source"],
        "recommendations_catalog_shades_considered": recommendation_result["catalog_shades_considered"],
        "recommendations_blocked": False,
        "recommendations_status": "ready",
        "recommendations_search_summary": recommendation_result["search_summary"],
        "recommendations_model": recommendation_result["model"],
        "recommendations_strategy": recommendation_result["strategy"],
        "recommendations_fallback_used": recommendation_result["fallback_used"],
        "recommendations_primary_error": recommendation_result["primary_error"],
    }


@app.get("/api/products/allergens")
def get_catalog_allergens():
    """Values the frontend can render as allergen-exclusion checkboxes."""
    return {"allergens": DEFAULT_ALLERGEN_OPTIONS}
