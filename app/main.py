import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.endpoints import transcription, health

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('transcription_api.log')
    ]
)

logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG,
    description="""
    API para transcripción automática de videos usando Whisper de OpenAI.
    
    ## Características
    - Upload de videos (.mp4, .avi, .mov, .mkv, .webm)
    - Transcripción automática con Whisper
    - Soporte multiidioma y traducción
    - Generación de subtítulos SRT de alta calidad
    - Evaluación automática de calidad
    - API REST completa
    
    ## Autenticación
    Incluye tu API key en el header Authorization: Bearer YOUR_API_KEY
    """
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(
    transcription.router,
    prefix="/api/v1",
    tags=["transcription"]
)

app.include_router(
    health.router,
    prefix="/api/v1",
    tags=["health"]
)

# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Endpoint raíz
@app.get("/")
async def root():
    return {
        "message": "Video Transcription API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Endpoint de información
@app.get("/info")
async def info():
    return {
        "api_title": settings.API_TITLE,
        "version": settings.API_VERSION,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "allowed_extensions": settings.ALLOWED_EXTENSIONS
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
        )
