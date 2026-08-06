"""Password-only site authentication backed by environment variables."""
import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel


router = APIRouter()
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class LoginRequest(BaseModel):
    password: str


def _site_password():
    password = os.getenv("SITE_PASSWORD")
    if not password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SITE_PASSWORD is not configured",
        )
    return password


def _secret_key():
    configured = os.getenv("JWT_SECRET_KEY")
    if configured:
        return configured
    # Derive a signing key without exposing SITE_PASSWORD in responses or logs.
    return hashlib.sha256(("kabumikke-session:" + _site_password()).encode("utf-8")).hexdigest()


def _expire_minutes():
    return max(5, int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720")))


def authenticate_password(password):
    return secrets.compare_digest(password.encode("utf-8"), _site_password().encode("utf-8"))


def create_access_token():
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=_expire_minutes())
    token = jwt.encode({"sub": "site-user", "iat": now, "exp": expires}, _secret_key(), algorithm=ALGORITHM)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": expires.isoformat(),
    }


def require_authenticated(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, _secret_key(), algorithms=[ALGORITHM])
        if payload.get("sub") != "site-user":
            raise JWTError("invalid subject")
        return {"authenticated": True}
    except JWTError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error


@router.post("/login")
async def login(request: LoginRequest):
    if not authenticate_password(request.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Password is incorrect")
    return create_access_token()


# Compatibility endpoint for clients using the OAuth2 form format.
@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_password(form_data.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Password is incorrect")
    return create_access_token()


@router.get("/session")
async def session(_: dict = Depends(require_authenticated)):
    return {"authenticated": True}
