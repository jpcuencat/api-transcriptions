from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import mimetypes

class FileValidator:
    """Validador robusto de archivos con manejo mejorado de edge cases."""
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 500) -> bool:
        """
        Valida el tamaño de archivo con mejor manejo de edge cases.
        
        Args:
            file_size: Tamaño del archivo en bytes
            max_size_mb: Tamaño máximo permitido en MB
            
        Returns:
            bool: True si el archivo es válido
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if max_size_mb <= 0:
            raise ValueError("max_size_mb debe ser mayor que 0")
        
        # Manejar tamaños negativos o inválidos
        if file_size < 0:
            return False
        
        # Archivo vacío es técnicamente válido pero puede no ser útil
        if file_size == 0:
            return True  # Permitir pero el usuario debería ser advertido
        
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """
        Valida la extensión del archivo con manejo robusto.
        
        Args:
            filename: Nombre del archivo
            allowed_extensions: Lista de extensiones permitidas (con punto, ej: ['.mp4'])
            
        Returns:
            bool: True si la extensión es válida
        """
        if not filename or not isinstance(filename, str):
            return False
        
        if not allowed_extensions:
            return False
        
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Normalizar extensiones permitidas a lowercase
            normalized_extensions = [ext.lower() for ext in allowed_extensions]
            
            return file_ext in normalized_extensions
        except Exception:
            return False
    
    @staticmethod
    def validate_file_type(file_path: str) -> bool:
        """
        Valida el tipo MIME del archivo usando mimetypes (más compatible).
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            bool: True si es un tipo de video válido
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # Usar mimetypes que es más estable y no requiere libmagic
            file_type, _ = mimetypes.guess_type(file_path)
            
            if not file_type:
                # Fallback: verificar por extensión
                ext = os.path.splitext(file_path)[1].lower()
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.m4v', '.flv']
                return ext in video_extensions
            
            video_types = [
                'video/mp4', 
                'video/avi', 
                'video/quicktime', 
                'video/x-msvideo', 
                'video/webm',
                'video/x-matroska',  # .mkv
                'video/x-ms-wmv'     # .wmv
            ]
            
            return file_type in video_types
        except Exception:
            # Fallback: verificar por extensión si falla todo lo demás
            try:
                ext = os.path.splitext(file_path)[1].lower()
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.m4v', '.flv']
                return ext in video_extensions
            except:
                return False
    
    @staticmethod
    def validate_filename_security(filename: str) -> bool:
        """
        Valida que el nombre de archivo sea seguro.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            bool: True si el nombre es seguro
        """
        if not filename:
            return False
        
        # Caracteres peligrosos
        dangerous_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
        
        for char in dangerous_chars:
            if char in filename:
                return False
        
        # No debe empezar con punto (archivos ocultos)
        if filename.startswith('.'):
            return False
        
        return True
    
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """
        Obtiene información detallada del archivo.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            dict: Información del archivo
        """
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'exists': True
            }
        except Exception:
            return {
                'size': 0,
                'created': None,
                'modified': None,
                'exists': False
            }
    
    @staticmethod
    async def validate_video_file(video_file, settings) -> dict:
        """
        Validación completa de archivo de video.
        
        Args:
            video_file: UploadFile de FastAPI
            settings: Configuración de la aplicación
            
        Returns:
            dict: Resultado de validación con detalles
            
        Raises:
            HTTPException: Si la validación falla
        """
        from fastapi import HTTPException
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        # Validar extensión
        if not FileValidator.validate_file_extension(video_file.filename, settings.ALLOWED_EXTENSIONS):
            validation_result['valid'] = False
            validation_result['errors'].append({
                "error_code": "INVALID_EXTENSION",
                "detail": f"File extension not supported. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
                "suggestions": [
                    "Convert your video to a supported format",
                    "Check that your file has the correct extension"
                ]
            })
        
        # Validar seguridad del nombre
        if not FileValidator.validate_filename_security(video_file.filename):
            validation_result['valid'] = False
            validation_result['errors'].append({
                "error_code": "UNSAFE_FILENAME", 
                "detail": "Filename contains unsafe characters",
                "suggestions": ["Rename your file with only letters, numbers, dots, and dashes"]
            })
        
        # Validar tamaño
        content = await video_file.read()
        file_size = len(content)
        await video_file.seek(0)  # Reset pointer
        
        if not FileValidator.validate_file_size(file_size, settings.MAX_FILE_SIZE_MB):
            validation_result['valid'] = False
            validation_result['errors'].append({
                "error_code": "FILE_TOO_LARGE",
                "detail": f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB",
                "suggestions": [
                    "Compress your video file",
                    "Upload a shorter video segment"
                ]
            })
        
        # Advertencias para archivos vacíos
        if file_size == 0:
            validation_result['warnings'].append({
                "warning_code": "EMPTY_FILE",
                "detail": "File appears to be empty",
                "suggestions": ["Ensure your video file has content"]
            })
        
        validation_result['file_info'] = {
            'filename': video_file.filename,
            'size': file_size,
            'content_type': video_file.content_type
        }
        
        return validation_result

class TranscriptionRequest(BaseModel):
    language: Optional[str] = Field(default="auto", description="Language code or 'auto' for detection")
    model_size: Optional[str] = Field(default="base", description="Whisper model size")
    translate_to: Optional[str] = Field(default=None, description="Target language for translation")
    quality_evaluation: Optional[bool] = Field(default=True, description="Enable quality evaluation")

class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

class QualityMetrics(BaseModel):
    confidence_score: float
    word_count: int
    speech_rate: float
    silence_ratio: float
    repetition_score: float
    language_consistency: float

class QualityReport(BaseModel):
    overall_score: float
    quality_level: str
    metrics: QualityMetrics
    recommendations: List[str]

class VideoInfo(BaseModel):
    duration: float
    size: int
    video_codec: Optional[str]
    audio_codec: Optional[str]
    fps: Optional[float]

class TranscriptionResult(BaseModel):
    job_id: str
    status: str
    video_info: Optional[VideoInfo] = None
    transcription_text: Optional[str] = None
    detected_language: Optional[str] = None
    segments: Optional[List[TranscriptionSegment]] = None
    quality_report: Optional[QualityReport] = None
    srt_file_path: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    job_id: Optional[str] = None
