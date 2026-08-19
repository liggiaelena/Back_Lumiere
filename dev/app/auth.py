import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timezone

from fastapi import HTTPException, status


JWT_SECRET = os.getenv("JWT_SECRET", "lumiere-development-secret-change-me")
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", "86400"))
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=32,
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64url_encode(salt)}${_b64url_encode(derived)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_b64url_decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=32,
        )
        return hmac.compare_digest(derived, _b64url_decode(expected))
    except (TypeError, ValueError):
        return False


def create_access_token(user_id: str) -> tuple[str, int]:
    now = int(time.time())
    expires_at = now + JWT_TTL_SECONDS
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url_encode(
        json.dumps({"sub": user_id, "iat": now, "exp": expires_at}).encode()
    )
    unsigned = f"{header}.{payload}"
    signature = hmac.new(
        JWT_SECRET.encode("utf-8"), unsigned.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{unsigned}.{_b64url_encode(signature)}", JWT_TTL_SECONDS


def decode_access_token(token: str) -> str:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired access token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        header, payload, signature = token.split(".")
        unsigned = f"{header}.{payload}"
        expected = hmac.new(
            JWT_SECRET.encode("utf-8"), unsigned.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _b64url_decode(signature)):
            raise unauthorized
        claims = json.loads(_b64url_decode(payload))
        if not claims.get("sub") or int(claims.get("exp", 0)) <= int(time.time()):
            raise unauthorized
        return str(claims["sub"])
    except HTTPException:
        raise
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise unauthorized


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
