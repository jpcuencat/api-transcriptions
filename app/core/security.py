import time
import logging
from typing import Dict, Optional
from fastapi import HTTPException, Depends, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

security = HTTPBearer(auto_error=False)

class SecurityManager:
    def __init__(self):
        self.api_keys = {
            settings.API_KEY: {
                "name": "Development User",
                "active": True
            }
        }
        self.rate_limits = {}  # Simple in-memory rate limiting
        self.max_requests_per_hour = 60
    
    async def verify_api_key(self, 
                           credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
                           x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> Dict:
        """Verifica API key - soporta Bearer Token y X-API-Key header"""
        api_key = None
        
        # Intentar obtener API key de Bearer token
        if credentials and credentials.credentials:
            api_key = credentials.credentials
            logging.info("Using Bearer token authentication")
        
        # Si no hay Bearer token, intentar X-API-Key header
        elif x_api_key:
            api_key = x_api_key
            logging.info("Using X-API-Key header authentication")
        
        # Si no hay ninguno, error de autenticación
        if not api_key:
            logging.warning("No authentication provided")
            raise HTTPException(
                status_code=401,
                detail="Not authenticated. Provide either 'Authorization: Bearer <token>' or 'X-API-Key: <key>' header"
            )
        
        # Verificar que la API key sea válida
        if api_key not in self.api_keys:
            logging.warning(f"Invalid API key attempted: {api_key}")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        user_info = self.api_keys[api_key]
        if not user_info.get("active", False):
            raise HTTPException(
                status_code=401,
                detail="API key is inactive"
            )
        
        return user_info
    
    async def check_rate_limit(self, request: Request) -> bool:
        """Implementa rate limiting simple"""
        client_ip = request.client.host
        current_time = time.time()
        
        # Limpiar entradas antiguas
        self._cleanup_rate_limits(current_time)
        
        # Verificar rate limit para esta IP
        if client_ip in self.rate_limits:
            requests = self.rate_limits[client_ip]
            recent_requests = [req_time for req_time in requests if current_time - req_time < 3600]
            
            if len(recent_requests) >= self.max_requests_per_hour:
                logging.warning(f"Rate limit exceeded for IP: {client_ip}")
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Maximum 60 requests per hour."
                )
            
            self.rate_limits[client_ip] = recent_requests + [current_time]
        else:
            self.rate_limits[client_ip] = [current_time]
        
        return True
    
    def _cleanup_rate_limits(self, current_time: float) -> None:
        """Limpia entradas de rate limit antiguas"""
        for ip in list(self.rate_limits.keys()):
            recent_requests = [req_time for req_time in self.rate_limits[ip] if current_time - req_time < 3600]
            if recent_requests:
                self.rate_limits[ip] = recent_requests
            else:
                del self.rate_limits[ip]

# Instancia global
security_manager = SecurityManager()
