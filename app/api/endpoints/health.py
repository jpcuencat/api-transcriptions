import os
import psutil
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Verificar servicios esenciales
        health_status = {
            "status": "healthy",
            "services": {
                "api": "running",
                "ffmpeg": check_ffmpeg(),
                "whisper": check_whisper(),
                "storage": check_storage()
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }
        
        # Determinar estado general
        failed_services = [k for k, v in health_status["services"].items() if v != "ok"]
        if failed_services:
            health_status["status"] = "degraded"
            health_status["failed_services"] = failed_services
        
        return health_status
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def check_ffmpeg() -> str:
    """Verifica si FFmpeg está disponible"""
    try:
        import subprocess
        # Verificar ffmpeg command line
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return "ok"
        else:
            return "error"
    except FileNotFoundError:
        return "not_installed"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "error"

def check_whisper() -> str:
    """Verifica si Whisper está disponible"""
    try:
        import whisper
        # Verificar si el modelo base está disponible
        model_path = os.path.join(settings.WHISPER_CACHE_DIR, "base.pt")
        if os.path.exists(model_path):
            return "ok"
        else:
            return "model_not_downloaded"
    except:
        return "error"

def check_storage() -> str:
    """Verifica si los directorios de almacenamiento están disponibles"""
    try:
        required_dirs = [
            settings.TEMP_DIR,
            f"{settings.TEMP_DIR}/uploads",
            f"{settings.TEMP_DIR}/audio",
            f"{settings.TEMP_DIR}/srt"
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                return f"missing_directory_{directory}"
            if not os.access(directory, os.W_OK):
                return f"no_write_access_{directory}"
        
        return "ok"
    except:
        return "error"
