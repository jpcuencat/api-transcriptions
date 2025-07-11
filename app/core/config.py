from pydantic_settings import BaseSettings
from typing import Optional
import os
import logging

class Settings(BaseSettings):
    # API Configuration
    API_TITLE: str = "Video Transcription API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # File Processing
    MAX_FILE_SIZE_MB: int = 500
    TEMP_DIR: str = "./temp"
    ALLOWED_EXTENSIONS: list = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    CLEANUP_TEMP_FILES: bool = True
    
    # Whisper Configuration
    WHISPER_MODEL_SIZE: str = "base"  # tiny, base, small, medium, large
    WHISPER_CACHE_DIR: str = "./temp/whisper_cache"
    WHISPER_DEVICE: str = "auto"  # auto, cpu, cuda
    
    # Security
    API_KEY: str = "dev_api_key_12345"
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # seconds
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "transcription_api.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_MAX_SIZE: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Performance
    MAX_CONCURRENT_JOBS: int = 3
    JOB_TIMEOUT_SECONDS: int = 3600  # 1 hour
    
    # Quality Evaluation
    ENABLE_QUALITY_EVALUATION: bool = True
    QUALITY_THRESHOLD_WARNING: float = 0.6
    
    # Supported Languages
    SUPPORTED_LANGUAGES: dict = {
        "auto": "Auto-detect",
        "es": "Spanish",
        "en": "English", 
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "ar": "Arabic"
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def setup_logging(settings: Settings):
    """Configura el sistema de logging mejorado."""
    import logging.handlers
    
    # Configurar nivel de logging
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Crear logger principal
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Formatter
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler con rotación
    if settings.LOG_FILE:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=settings.LOG_MAX_SIZE,
            backupCount=settings.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Configurar loggers específicos
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logger

# Crear instancia global
settings = Settings()

# Configurar logging al inicio
setup_logging(settings)

# Crear directorios necesarios
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(f"{settings.TEMP_DIR}/uploads", exist_ok=True)
os.makedirs(f"{settings.TEMP_DIR}/audio", exist_ok=True)
os.makedirs(f"{settings.TEMP_DIR}/srt", exist_ok=True)
os.makedirs(settings.WHISPER_CACHE_DIR, exist_ok=True)

# Log configuración inicial
logger = logging.getLogger(__name__)
logger.info(f"API Configuration loaded - Debug: {settings.DEBUG}, Log Level: {settings.LOG_LEVEL}")
logger.info(f"Temp directory: {settings.TEMP_DIR}")
logger.info(f"Max file size: {settings.MAX_FILE_SIZE_MB}MB")
logger.info(f"Whisper model: {settings.WHISPER_MODEL_SIZE}, Device: {settings.WHISPER_DEVICE}")
