import os
import uuid
import shutil
import logging
import aiofiles
from typing import Optional
from fastapi import UploadFile
from app.core.config import settings

class FileHandler:
    def __init__(self):
        self.temp_dir = settings.TEMP_DIR
        self.uploads_dir = f"{self.temp_dir}/uploads"
        self.audio_dir = f"{self.temp_dir}/audio"
        self.srt_dir = f"{self.temp_dir}/srt"
        
        # Crear directorios si no existen
        for directory in [self.uploads_dir, self.audio_dir, self.srt_dir]:
            os.makedirs(directory, exist_ok=True)
    
    async def save_uploaded_file(self, file: UploadFile, job_id: str) -> str:
        """Guarda archivo subido y retorna path"""
        try:
            # Sanitizar nombre de archivo
            safe_filename = self._sanitize_filename(file.filename)
            file_path = f"{self.uploads_dir}/{job_id}_{safe_filename}"
            
            logging.info(f"Saving uploaded file: {file_path}")
            
            async with aiofiles.open(file_path, 'wb') as buffer:
                content = await file.read()
                await buffer.write(content)
            
            # Verificar que el archivo se guardó correctamente
            if not os.path.exists(file_path):
                raise Exception("File was not saved correctly")
            
            file_size = os.path.getsize(file_path)
            logging.info(f"File saved successfully: {file_path} ({file_size} bytes)")
            
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving uploaded file: {e}")
            raise Exception(f"Failed to save uploaded file: {e}")
    
    def generate_audio_path(self, job_id: str) -> str:
        """Genera path para archivo de audio extraído"""
        return f"{self.audio_dir}/{job_id}.wav"
    
    def generate_srt_path(self, job_id: str) -> str:
        """Genera path para archivo SRT"""
        return f"{self.srt_dir}/{job_id}.srt"
    
    def cleanup_job_files(self, job_id: str) -> None:
        """Limpia archivos temporales de un trabajo"""
        try:
            patterns = [
                f"{self.uploads_dir}/{job_id}_*",
                f"{self.audio_dir}/{job_id}.*",
                # Mantener SRT files para descarga
            ]
            
            for pattern in patterns:
                import glob
                for file_path in glob.glob(pattern):
                    try:
                        os.remove(file_path)
                        logging.info(f"Cleaned up file: {file_path}")
                    except Exception as e:
                        logging.warning(f"Could not remove {file_path}: {e}")
                        
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        """Limpia archivos antiguos"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for directory in [self.uploads_dir, self.audio_dir, self.srt_dir]:
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > max_age_seconds:
                            os.remove(file_path)
                            logging.info(f"Removed old file: {file_path}")
                            
        except Exception as e:
            logging.error(f"Error cleaning old files: {e}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitiza nombre de archivo"""
        # Remover caracteres peligrosos
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        sanitized = "".join(c for c in filename if c in safe_chars)
        
        # Asegurar que no esté vacío
        if not sanitized:
            sanitized = "video.mp4"
        
        # Limitar longitud
        if len(sanitized) > 100:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:95] + ext
        
        return sanitized
    
    def get_file_size(self, file_path: str) -> int:
        """Obtiene tamaño de archivo"""
        return os.path.getsize(file_path) if os.path.exists(file_path) else 0
    
    def file_exists(self, file_path: str) -> bool:
        """Verifica si archivo existe"""
        return os.path.exists(file_path)
