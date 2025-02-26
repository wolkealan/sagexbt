from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional, Callable
import time
from datetime import datetime, timedelta
import json
import hashlib
import hmac

from config.config import APIConfig
from utils.logger import get_api_logger

logger = get_api_logger()

security = HTTPBearer(auto_error=False)

class APIMiddleware:
    """Middleware for API request processing and security"""
    
    @staticmethod
    async def log_request(request: Request, call_next):
        """Log incoming API requests"""
        start_time = time.time()
        
        # Log request details
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)
        
        # Allow request to be processed
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response details
        status_code = response.status_code
        
        logger.info(f"{client_ip} - {method} {url} - {status_code} - {process_time:.4f}s")
        
        return response
    
    @staticmethod
    async def rate_limiter(request: Request, call_next, 
                          max_requests: int = 100, window_seconds: int = 60):
        """
        Rate limit API requests
        
        Args:
            request: FastAPI request object
            call_next: Next middleware function
            max_requests: Maximum requests allowed per window
            window_seconds: Time window in seconds
            
        Returns:
            FastAPI response
        """
        # Get client identifier (IP or API key if available)
        client_id = APIMiddleware._get_client_identifier(request)
        
        # Check if client has exceeded rate limit
        if APIMiddleware._is_rate_limited(client_id, max_requests, window_seconds):
            logger.warning(f"Rate limit exceeded for {client_id}")
            return Response(
                content=json.dumps({"error": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json"
            )
        
        # Process the request
        response = await call_next(request)
        return response
    
    @staticmethod
    def _get_client_identifier(request: Request) -> str:
        """Get a unique identifier for the client"""
        # Try to get API key from header or query param
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            # Use client IP as fallback
            return request.client.host if request.client else "unknown"
        
        return api_key
    
    @staticmethod
    def _is_rate_limited(client_id: str, max_requests: int, window_seconds: int) -> bool:
        """
        Check if client has exceeded rate limits
        
        Note: In a production environment, this would use Redis or similar for
        distributed rate limiting. This is a simplified in-memory implementation.
        """
        # Static variable for request tracking
        if not hasattr(APIMiddleware._is_rate_limited, "request_log"):
            APIMiddleware._is_rate_limited.request_log = {}
        
        # Get current time
        current_time = time.time()
        
        # Clean up old entries
        for cid in list(APIMiddleware._is_rate_limited.request_log.keys()):
            entries = APIMiddleware._is_rate_limited.request_log[cid]
            # Remove entries older than the window
            APIMiddleware._is_rate_limited.request_log[cid] = [
                entry for entry in entries if entry > current_time - window_seconds
            ]
            # Remove client if no entries left
            if not APIMiddleware._is_rate_limited.request_log[cid]:
                del APIMiddleware._is_rate_limited.request_log[cid]
        
        # Get client's request history
        client_requests = APIMiddleware._is_rate_limited.request_log.get(client_id, [])
        
        # Check if client has exceeded limit
        if len(client_requests) >= max_requests:
            return True
        
        # Add current request timestamp
        if client_id not in APIMiddleware._is_rate_limited.request_log:
            APIMiddleware._is_rate_limited.request_log[client_id] = []
        APIMiddleware._is_rate_limited.request_log[client_id].append(current_time)
        
        return False
    
    @staticmethod
    async def error_handler(request: Request, call_next):
        """Global error handler for API requests"""
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            
            # Return a proper error response
            return Response(
                content=json.dumps({
                    "error": "Internal Server Error",
                    "message": str(e) if APIConfig.DEBUG else "An unexpected error occurred"
                }),
                status_code=500,
                media_type="application/json"
            )
    
    @staticmethod
    async def auth_required(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
        """
        Verify API key for protected endpoints
        
        Args:
            request: FastAPI request object
            credentials: HTTP Authorization credentials
            
        Returns:
            API key if valid
            
        Raises:
            HTTPException if invalid or missing
        """
        if not credentials:
            # Check for API key in header
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail="API key is required"
                )
        else:
            # Use token as API key
            api_key = credentials.credentials
        
        # Verify API key
        if not APIMiddleware._is_valid_api_key(api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        return api_key
    
    @staticmethod
    def _is_valid_api_key(api_key: str) -> bool:
        """
        Validate API key
        
        Note: In a production environment, this would check against a database
        of valid API keys. This is a simplified implementation.
        """
        # For demo purposes, accept any non-empty string
        # In production, replace with proper validation
        return bool(api_key and len(api_key) > 8)
    
    @staticmethod
    async def validate_webhook_signature(request: Request, call_next):
        """Validate webhook signatures for webhook endpoints"""
        # Only process webhook endpoints
        if not request.url.path.startswith("/webhook"):
            return await call_next(request)
        
        # Get signature from header
        signature = request.headers.get("X-Signature")
        if not signature:
            logger.warning("Missing webhook signature")
            return Response(
                content=json.dumps({"error": "Missing signature"}),
                status_code=401,
                media_type="application/json"
            )
        
        # Get request body
        body = await request.body()
        
        # Validate signature
        if not APIMiddleware._verify_signature(body, signature):
            logger.warning("Invalid webhook signature")
            return Response(
                content=json.dumps({"error": "Invalid signature"}),
                status_code=401,
                media_type="application/json"
            )
        
        # Signature is valid, process the request
        return await call_next(request)
    
    @staticmethod
    def _verify_signature(body: bytes, signature: str) -> bool:
        """
        Verify webhook signature
        
        Args:
            body: Request body bytes
            signature: Signature string from header
            
        Returns:
            True if signature is valid, False otherwise
        """
        # In a real implementation, use a secure webhook secret
        webhook_secret = "your_webhook_secret"
        
        # Calculate expected signature
        expected_signature = hmac.new(
            webhook_secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(signature, expected_signature)

# Dependency for requiring authentication
def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get and validate API key from request"""
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="API key is required"
        )
    
    api_key = credentials.credentials
    
    # Verify API key
    if not APIMiddleware._is_valid_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key

# Register middleware functions with FastAPI
def register_middleware(app):
    """Register all middleware functions with the FastAPI app"""
    # Add global error handler (should be first)
    app.middleware("http")(APIMiddleware.error_handler)
    
    # Add request logger
    app.middleware("http")(APIMiddleware.log_request)
    
    # Add rate limiter
    app.middleware("http")(lambda req, call_next: APIMiddleware.rate_limiter(req, call_next))
    
    # Add webhook signature validation
    app.middleware("http")(APIMiddleware.validate_webhook_signature)
    
    logger.info("Registered API middleware functions")