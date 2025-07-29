"""
Authentication service for user management.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging

from src.app.models.models import User, UserInDB
from src.app.utils.error_handling import AuthError
from src.app.utils.caching import cache_result

# JWT settings
SECRET_KEY = "your-secret-key"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    """Service for authentication and user management."""
    
    def __init__(self):
        self.users_db = {}  # In-memory store, replace with database
        self.token_blacklist = set()
    
    async def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        try:
            to_encode = data.copy()
            
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(
                    minutes=ACCESS_TOKEN_EXPIRE_MINUTES
                )
            
            to_encode.update({
                "exp": expire,
                "iat": datetime.utcnow()
            })
            
            encoded_jwt = jwt.encode(
                to_encode,
                SECRET_KEY,
                algorithm=ALGORITHM
            )
            
            return encoded_jwt
            
        except Exception as e:
            logging.error(f"Token creation failed: {str(e)}")
            raise AuthError(
                "Failed to create access token",
                details={"error": str(e)}
            )
    
    @cache_result(expire=300)  # Cache for 5 minutes
    async def get_user(
        self,
        username: str
    ) -> Optional[UserInDB]:
        """Get user from database."""
        try:
            if username in self.users_db:
                user_dict = self.users_db[username]
                return UserInDB(**user_dict)
            return None
            
        except Exception as e:
            logging.error(f"User retrieval failed: {str(e)}")
            raise AuthError(
                "Failed to get user",
                details={"error": str(e)}
            )
    
    async def create_user(
        self,
        username: str,
        password: str
    ) -> UserInDB:
        """Create new user."""
        try:
            if username in self.users_db:
                raise AuthError(
                    "Username already exists",
                    details={"username": username}
                )
            
            hashed_password = self.get_password_hash(password)
            
            user_dict = {
                "username": username,
                "hashed_password": hashed_password,
                "created_at": datetime.utcnow().isoformat(),
                "is_active": True
            }
            
            self.users_db[username] = user_dict
            return UserInDB(**user_dict)
            
        except Exception as e:
            logging.error(f"User creation failed: {str(e)}")
            raise AuthError(
                "Failed to create user",
                details={"error": str(e)}
            )
    
    async def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[UserInDB]:
        """Authenticate user with username and password."""
        try:
            user = await self.get_user(username)
            if not user:
                return None
            
            if not self.verify_password(password, user.hashed_password):
                return None
            
            return user
            
        except Exception as e:
            logging.error(f"Authentication failed: {str(e)}")
            raise AuthError(
                "Failed to authenticate user",
                details={"error": str(e)}
            )
    
    def verify_password(
        self,
        plain_password: str,
        hashed_password: str
    ) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(
        self,
        password: str
    ) -> str:
        """Get password hash."""
        return pwd_context.hash(password)
    
    async def update_password(
        self,
        username: str,
        new_password: str
    ) -> bool:
        """Update user password."""
        try:
            user = await self.get_user(username)
            if not user:
                raise AuthError(
                    "User not found",
                    details={"username": username}
                )
            
            hashed_password = self.get_password_hash(new_password)
            self.users_db[username]["hashed_password"] = hashed_password
            
            return True
            
        except Exception as e:
            logging.error(f"Password update failed: {str(e)}")
            raise AuthError(
                "Failed to update password",
                details={"error": str(e)}
            )
    
    async def invalidate_token(
        self,
        username: str
    ) -> bool:
        """Invalidate user's current token."""
        try:
            self.token_blacklist.add(username)
            return True
            
        except Exception as e:
            logging.error(f"Token invalidation failed: {str(e)}")
            raise AuthError(
                "Failed to invalidate token",
                details={"error": str(e)}
            )
    
    @staticmethod
    async def get_current_user(token: str) -> User:
        """Get current user from token."""
        try:
            payload = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=[ALGORITHM]
            )
            
            username = payload.get("sub")
            if username is None:
                raise AuthError("Could not validate credentials")
            
            user = await AuthService().get_user(username)
            if user is None:
                raise AuthError("User not found")
            
            if username in AuthService().token_blacklist:
                raise AuthError("Token has been invalidated")
            
            return user
            
        except JWTError:
            raise AuthError("Could not validate credentials")
        except Exception as e:
            logging.error(f"Current user retrieval failed: {str(e)}")
            raise AuthError(
                "Failed to get current user",
                details={"error": str(e)}
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get authentication service status."""
        return {
            "active_users": len(self.users_db),
            "blacklisted_tokens": len(self.token_blacklist),
            "timestamp": datetime.utcnow().isoformat()
        } 