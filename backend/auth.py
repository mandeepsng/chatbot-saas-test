"""
Authentication and Authorization System for ChatFlow AI
Handles JWT tokens, API keys, rate limiting, and user permissions
"""

import os
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from functools import wraps
import asyncio
import time
from dataclasses import dataclass
from enum import Enum

import asyncpg
import redis
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token scheme
security = HTTPBearer(auto_error=False)

class UserRole(Enum):
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class PlanType(Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

@dataclass
class RateLimits:
    requests_per_hour: int
    requests_per_minute: int
    upload_size_mb: int
    training_jobs_per_day: int

# Plan-based rate limits
PLAN_LIMITS = {
    PlanType.STARTER: RateLimits(
        requests_per_hour=1000,
        requests_per_minute=50,
        upload_size_mb=10,
        training_jobs_per_day=5
    ),
    PlanType.PROFESSIONAL: RateLimits(
        requests_per_hour=10000,
        requests_per_minute=200,
        upload_size_mb=100,
        training_jobs_per_day=20
    ),
    PlanType.ENTERPRISE: RateLimits(
        requests_per_hour=100000,
        requests_per_minute=1000,
        upload_size_mb=1000,
        training_jobs_per_day=100
    )
}

class TokenData(BaseModel):
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[str] = None
    plan_type: Optional[str] = None
    scopes: List[str] = []

class User(BaseModel):
    id: str
    email: str
    username: Optional[str] = None
    role: UserRole
    plan_type: PlanType
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    api_key_hash: Optional[str] = None

class AuthService:
    def __init__(self, db_config: Dict[str, str], redis_client: redis.Redis):
        self.db_config = db_config
        self.redis = redis_client
        
    async def get_db_connection(self):
        return await asyncpg.connect(**self.db_config)
    
    # Password utilities
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return f"cf_{secrets.token_urlsafe(32)}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    # JWT Token utilities
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
            
            token_data = TokenData(
                user_id=user_id,
                username=payload.get("username"),
                role=payload.get("role"),
                plan_type=payload.get("plan_type"),
                scopes=payload.get("scopes", [])
            )
            return token_data
        except jwt.PyJWTError:
            return None
    
    # User management
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        conn = await self.get_db_connection()
        try:
            user_data = await conn.fetchrow("""
                SELECT id, email, username, password_hash, role, plan_type, 
                       is_active, created_at, last_login, api_key_hash
                FROM users 
                WHERE email = $1 AND is_active = true
            """, email)
            
            if not user_data:
                return None
            
            if not self.verify_password(password, user_data['password_hash']):
                return None
            
            # Update last login
            await conn.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = $1
            """, user_data['id'])
            
            return User(
                id=user_data['id'],
                email=user_data['email'],
                username=user_data['username'],
                role=UserRole(user_data['role']),
                plan_type=PlanType(user_data['plan_type']),
                is_active=user_data['is_active'],
                created_at=user_data['created_at'],
                last_login=user_data['last_login'],
                api_key_hash=user_data['api_key_hash']
            )
            
        finally:
            await conn.close()
    
    async def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user with API key"""
        api_key_hash = self.hash_api_key(api_key)
        
        conn = await self.get_db_connection()
        try:
            user_data = await conn.fetchrow("""
                SELECT id, email, username, role, plan_type, 
                       is_active, created_at, last_login, api_key_hash
                FROM users 
                WHERE api_key_hash = $1 AND is_active = true
            """, api_key_hash)
            
            if not user_data:
                return None
            
            return User(
                id=user_data['id'],
                email=user_data['email'],
                username=user_data['username'],
                role=UserRole(user_data['role']),
                plan_type=PlanType(user_data['plan_type']),
                is_active=user_data['is_active'],
                created_at=user_data['created_at'],
                last_login=user_data['last_login'],
                api_key_hash=user_data['api_key_hash']
            )
            
        finally:
            await conn.close()
    
    async def get_current_user(self, token: str) -> Optional[User]:
        """Get current user from token"""
        token_data = self.verify_token(token)
        if not token_data or not token_data.user_id:
            return None
        
        conn = await self.get_db_connection()
        try:
            user_data = await conn.fetchrow("""
                SELECT id, email, username, role, plan_type, 
                       is_active, created_at, last_login, api_key_hash
                FROM users 
                WHERE id = $1 AND is_active = true
            """, token_data.user_id)
            
            if not user_data:
                return None
            
            return User(
                id=user_data['id'],
                email=user_data['email'],
                username=user_data['username'],
                role=UserRole(user_data['role']),
                plan_type=PlanType(user_data['plan_type']),
                is_active=user_data['is_active'],
                created_at=user_data['created_at'],
                last_login=user_data['last_login'],
                api_key_hash=user_data['api_key_hash']
            )
            
        finally:
            await conn.close()
    
    async def create_user(self, email: str, password: str, username: Optional[str] = None,
                         role: UserRole = UserRole.USER, plan_type: PlanType = PlanType.STARTER) -> User:
        """Create a new user"""
        conn = await self.get_db_connection()
        try:
            # Check if user already exists
            existing = await conn.fetchval("SELECT id FROM users WHERE email = $1", email)
            if existing:
                raise ValueError("User with this email already exists")
            
            user_id = f"user_{secrets.token_urlsafe(16)}"
            password_hash = self.get_password_hash(password)
            api_key = self.generate_api_key()
            api_key_hash = self.hash_api_key(api_key)
            
            await conn.execute("""
                INSERT INTO users (id, email, username, password_hash, role, plan_type, 
                                 is_active, api_key_hash, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
            """, user_id, email, username, password_hash, role.value, plan_type.value, 
                True, api_key_hash)
            
            user = User(
                id=user_id,
                email=email,
                username=username,
                role=role,
                plan_type=plan_type,
                is_active=True,
                created_at=datetime.utcnow(),
                api_key_hash=api_key_hash
            )
            
            return user, api_key
            
        finally:
            await conn.close()

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_rate_limited(self, key: str, limit: int, window_seconds: int) -> bool:
        """Check if rate limit is exceeded"""
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_seconds)
            results = pipe.execute()
            
            current_count = results[0]
            return current_count > limit
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return False  # Allow on error
    
    async def get_rate_limit_info(self, key: str) -> Dict[str, int]:
        """Get current rate limit status"""
        try:
            current_count = await self.redis.get(key)
            ttl = await self.redis.ttl(key)
            
            return {
                'current': int(current_count) if current_count else 0,
                'ttl': ttl if ttl > 0 else 0
            }
        except Exception as e:
            logger.error(f"Rate limit info error: {e}")
            return {'current': 0, 'ttl': 0}

# Initialize services
auth_service = None
rate_limiter = None

def init_auth(db_config: Dict[str, str], redis_client: redis.Redis):
    global auth_service, rate_limiter
    auth_service = AuthService(db_config, redis_client)
    rate_limiter = RateLimiter(redis_client)

# Dependency functions
async def get_current_user_from_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Get current user from Bearer token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if it's an API key (starts with cf_)
    if credentials.credentials.startswith("cf_"):
        user = await auth_service.authenticate_api_key(credentials.credentials)
    else:
        # JWT token
        user = await auth_service.get_current_user(credentials.credentials)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user_from_token)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user

def require_role(required_role: UserRole):
    """Decorator to require specific user role"""
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        role_hierarchy = {
            UserRole.USER: 1,
            UserRole.ADMIN: 2,
            UserRole.SUPER_ADMIN: 3
        }
        
        if role_hierarchy.get(current_user.role, 0) < role_hierarchy.get(required_role, 99):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return role_checker

def require_plan(required_plan: PlanType):
    """Decorator to require minimum plan type"""
    def plan_checker(current_user: User = Depends(get_current_active_user)) -> User:
        plan_hierarchy = {
            PlanType.STARTER: 1,
            PlanType.PROFESSIONAL: 2,
            PlanType.ENTERPRISE: 3
        }
        
        if plan_hierarchy.get(current_user.plan_type, 0) < plan_hierarchy.get(required_plan, 99):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Requires {required_plan.value} plan or higher"
            )
        return current_user
    
    return plan_checker

def rate_limit(requests_per_minute: int = 60, requests_per_hour: int = 1000):
    """Rate limiting decorator"""
    def rate_limit_decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user from dependencies
            current_user = None
            for arg in args:
                if isinstance(arg, User):
                    current_user = arg
                    break
            
            if not current_user:
                # Try to extract from kwargs
                current_user = kwargs.get('current_user')
            
            if current_user:
                # User-specific rate limiting based on plan
                plan_limits = PLAN_LIMITS.get(current_user.plan_type)
                if plan_limits:
                    requests_per_minute = min(requests_per_minute, plan_limits.requests_per_minute)
                    requests_per_hour = min(requests_per_hour, plan_limits.requests_per_hour)
                
                user_key = f"rate_limit:user:{current_user.id}"
            else:
                # IP-based rate limiting for unauthenticated requests
                # This would need to be implemented with request object
                user_key = "rate_limit:anonymous"
            
            # Check minute limit
            minute_key = f"{user_key}:minute:{int(time.time() // 60)}"
            if await rate_limiter.is_rate_limited(minute_key, requests_per_minute, 60):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded - too many requests per minute"
                )
            
            # Check hour limit
            hour_key = f"{user_key}:hour:{int(time.time() // 3600)}"
            if await rate_limiter.is_rate_limited(hour_key, requests_per_hour, 3600):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded - too many requests per hour"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return rate_limit_decorator

# Login/logout endpoints
class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class RefreshRequest(BaseModel):
    refresh_token: str

async def login(login_data: LoginRequest) -> LoginResponse:
    """Login endpoint"""
    user = await auth_service.authenticate_user(login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Create tokens
    token_data = {
        "sub": user.id,
        "username": user.username,
        "role": user.role.value,
        "plan_type": user.plan_type.value,
        "scopes": ["read", "write"]
    }
    
    access_token = auth_service.create_access_token(token_data)
    refresh_token = auth_service.create_refresh_token({"sub": user.id})
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "role": user.role.value,
            "plan_type": user.plan_type.value
        }
    )

async def refresh_token(refresh_data: RefreshRequest) -> LoginResponse:
    """Refresh token endpoint"""
    token_data = auth_service.verify_token(refresh_data.refresh_token)
    if not token_data or not token_data.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = await auth_service.get_current_user(refresh_data.refresh_token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new tokens
    new_token_data = {
        "sub": user.id,
        "username": user.username,
        "role": user.role.value,
        "plan_type": user.plan_type.value,
        "scopes": ["read", "write"]
    }
    
    access_token = auth_service.create_access_token(new_token_data)
    new_refresh_token = auth_service.create_refresh_token({"sub": user.id})
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "role": user.role.value,
            "plan_type": user.plan_type.value
        }
    )

# Health check for auth
async def auth_health_check() -> Dict[str, str]:
    """Health check for authentication service"""
    try:
        # Test database connection
        conn = await auth_service.get_db_connection()
        await conn.fetchval("SELECT 1")
        await conn.close()
        
        # Test Redis connection
        await rate_limiter.redis.ping()
        
        return {"status": "healthy", "service": "auth"}
        
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}