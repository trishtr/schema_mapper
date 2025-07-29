"""
Router for authentication endpoints.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.app.models.models import User, UserInDB, Token, TokenData
from src.app.services.auth.auth_service import AuthService

router = APIRouter(
    prefix="/auth",
    tags=["authentication"]
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def get_auth_service() -> AuthService:
    """Dependency for auth service."""
    return AuthService()

@router.post(
    "/token",
    response_model=Token,
    summary="Create access token",
    description="Create access token for authenticated user."
)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, str]:
    """Login endpoint to get access token."""
    try:
        user = await auth_service.authenticate_user(
            form_data.username,
            form_data.password
        )
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        access_token = await auth_service.create_access_token(
            data={"sub": user.username}
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post(
    "/register",
    response_model=User,
    summary="Register new user",
    description="Register a new user with username and password."
)
async def register_user(
    username: str,
    password: str,
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """Register new user endpoint."""
    try:
        # Check if user exists
        existing_user = await auth_service.get_user(username)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Username already registered"
            )
        
        # Create new user
        user = await auth_service.create_user(
            username=username,
            password=password
        )
        
        return {
            "username": user.username,
            "created_at": user.created_at
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get(
    "/me",
    response_model=User,
    summary="Get current user",
    description="Get details of currently authenticated user."
)
async def read_users_me(
    current_user: User = Depends(AuthService.get_current_user)
) -> Dict[str, Any]:
    """Get current user endpoint."""
    return current_user

@router.post(
    "/change-password",
    summary="Change password",
    description="Change password for authenticated user."
)
async def change_password(
    old_password: str,
    new_password: str,
    current_user: User = Depends(AuthService.get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """Change password endpoint."""
    try:
        # Verify old password
        if not await auth_service.verify_password(
            old_password,
            current_user.hashed_password
        ):
            raise HTTPException(
                status_code=400,
                detail="Incorrect password"
            )
        
        # Update password
        await auth_service.update_password(
            current_user.username,
            new_password
        )
        
        return {
            "message": "Password updated successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post(
    "/logout",
    summary="Logout user",
    description="Logout current user and invalidate token."
)
async def logout(
    current_user: User = Depends(AuthService.get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """Logout endpoint."""
    try:
        await auth_service.invalidate_token(current_user.username)
        
        return {
            "message": "Successfully logged out",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 