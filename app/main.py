import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.endpoints import transcription, health, realtime

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
    API de Transcripción de Video/Audio con soporte de Tiempo Real
    
    Funcionalidades:
    - Transcripción de videos (.mp4, .avi, .mov, .mkv, .webm)
    - Transcripción de URLs (YouTube, Vimeo, etc.)
    - Transcripción de archivos de audio (.mp3, .wav, .flac, .aac, .ogg, .m4a, .wma)
    - **NUEVO**: Transcripción en tiempo real desde micrófono
    - Traducción automática a múltiples idiomas
    - Generación de archivos SRT
    - Evaluación de calidad de transcripción
    
    Funcionalidades de tiempo real:
    - Captura de audio desde micrófono del navegador
    - Transcripción en vivo mientras habla el usuario
    - Traducción simultánea en tiempo real
    - WebSocket para comunicación bidireccional
    - Detección de actividad de voz (VAD)
    """,
    contact={
        "name": "API Support",
        "email": "support@transcription-api.com"
    }
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(
    transcription.router,
    prefix="/api/v1/transcriptions",
    tags=["transcriptions"]
)

app.include_router(
    realtime.router,
    prefix="/api/v1/realtime",
    tags=["realtime"]
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
        "message": "Video Transcription API with Real-Time Support",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "features": {
            "video_transcription": "/api/v1/transcriptions/transcribe",
            "url_transcription": "/api/v1/transcriptions/transcribe-url", 
            "audio_transcription": "/api/v1/transcriptions/transcribe-audio",
            "realtime_transcription": "/api/v1/realtime/create-session",
            "websocket_realtime": "/api/v1/realtime/ws/{session_id}",
            "demo_page": "/realtime_demo.html"
        }
    }

# Endpoint de información
@app.get("/info")
async def info():
    return {
        "api_title": settings.API_TITLE,
        "version": settings.API_VERSION,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "allowed_video_extensions": settings.ALLOWED_EXTENSIONS,
        "allowed_audio_extensions": getattr(settings, 'ALLOWED_AUDIO_EXTENSIONS', []),
        "realtime_features": {
            "microphone_capture": True,
            "live_transcription": True,
            "real_time_translation": True,
            "websocket_support": True,
            "voice_activity_detection": True,
            "supported_audio_formats": ["webm", "mp3", "wav"],
            "chunk_duration": "1-10 seconds",
            "models": ["tiny", "base", "small"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
