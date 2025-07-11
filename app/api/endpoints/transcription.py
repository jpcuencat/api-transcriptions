import os
import time
import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import FileResponse

from app.models.schemas import TranscriptionRequest, TranscriptionResult, ErrorResponse
from app.utils.validators import FileValidator
from app.utils.file_handler import FileHandler
from app.services.audio_extractor import AudioExtractor
from app.services.transcription_service import TranscriptionService
from app.services.subtitle_generator import SubtitleGenerator
from app.services.quality_evaluator import QualityEvaluator
from app.services.translation_service import TranslationService
from app.core.security import security_manager
from app.core.config import settings

router = APIRouter()

# Instancias de servicios
file_validator = FileValidator()
file_handler = FileHandler()
audio_extractor = AudioExtractor()
transcription_service = TranscriptionService()
subtitle_generator = SubtitleGenerator()
quality_evaluator = QualityEvaluator()
translation_service = TranslationService()

# Almacenamiento simple en memoria para el estado de trabajos (en producción usar base de datos)
job_storage: dict = {}

@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_video(
    request: Request,
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    language: Optional[str] = Form("auto"),
    model_size: Optional[str] = Form("base"),
    translate_to: Optional[str] = Form(None),
    quality_evaluation: Optional[bool] = Form(True),
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """
    CLAUDE TEST: THIS ENDPOINT ALWAYS RETURNS SPANISH TRANSLATION NOW
    
    - **video_file**: Video file (.mp4, .avi, .mov, .mkv, .webm)
    - **language**: Language code or 'auto' for detection
    - **model_size**: Whisper model size (tiny, base, small, medium, large)
    - **translate_to**: Target language for translation (optional)
    - **quality_evaluation**: Enable quality evaluation (default: true)
    """
    job_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log de parámetros recibidos
    logging.info(f"Endpoint received parameters - language: {language}, model_size: {model_size}, translate_to: {translate_to}")
    
    try:
        # Verificar rate limit
        await security_manager.check_rate_limit(request)
        
        # Validar parámetros
        if language not in settings.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {language}. Supported: {list(settings.SUPPORTED_LANGUAGES.keys())}"
            )
        
        if model_size not in transcription_service.get_available_models():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model size: {model_size}. Available: {transcription_service.get_available_models()}"
            )
        
        if translate_to and translate_to not in settings.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported translation target: {translate_to}"
            )
        
        # Validar archivo con respuesta detallada
        validation_result = await file_validator.validate_video_file(video_file, settings)
        
        if not validation_result['valid']:
            # Combinar todos los errores en una respuesta estructurada
            error_details = {
                "error": "File Validation Failed",
                "detail": "One or more validation errors occurred",
                "validation_errors": validation_result['errors'],
                "timestamp": datetime.now().isoformat()
            }
            raise HTTPException(status_code=422, detail=error_details)
        
        # Log warnings if any
        if validation_result['warnings']:
            logging.warning(f"File validation warnings for {video_file.filename}: {validation_result['warnings']}")
        
        # Crear entrada de trabajo
        job_storage[job_id] = TranscriptionResult(
            job_id=job_id,
            status="processing",
            created_at=datetime.now()
        )
        
        logging.info(f"Starting transcription job {job_id} for user {user_info.get('name')}")
        
        # Procesar en background (para desarrollo, procesamos sincrónicamente)
        background_tasks.add_task(
            process_video_transcription,
            job_id=job_id,
            video_file=video_file,
            language=language,
            model_size=model_size,
            translate_to=translate_to,
            quality_evaluation=quality_evaluation,
            start_time=start_time
        )
        
        return job_storage[job_id]
        
    except HTTPException:
        raise
    except ValueError as e:
        logging.error(f"Validation error in transcription job: {e}")
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "Validation Error",
                "detail": str(e),
                "error_code": "VALIDATION_FAILED",
                "suggestions": ["Check file size and format", "Ensure file is a valid video"]
            }
        )
    except Exception as e:
        logging.error(f"Error starting transcription job: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal Server Error",
                "detail": f"Failed to start transcription: {str(e)}",
                "error_code": "PROCESSING_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )

@router.get("/transcribe/{job_id}/status", response_model=TranscriptionResult)
async def get_transcription_status(
    job_id: str,
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """Get transcription job status"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_storage[job_id]

@router.get("/transcribe/{job_id}/download")
async def download_srt_file(
    job_id: str,
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """Download generated SRT file"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Transcription not completed. Status: {job.status}"
        )
    
    if not job.srt_file_path or not os.path.exists(job.srt_file_path):
        raise HTTPException(status_code=404, detail="SRT file not found")
    
    return FileResponse(
        path=job.srt_file_path,
        filename=f"subtitles_{job_id}.srt",
        media_type="application/x-subrip"
    )

@router.delete("/transcribe/{job_id}")
async def delete_transcription_job(
    job_id: str,
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """Delete transcription job and cleanup files"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Limpiar archivos
    file_handler.cleanup_job_files(job_id)
    
    # Remover de storage
    del job_storage[job_id]
    
    return {"message": "Job deleted successfully"}

@router.get("/languages")
async def get_supported_languages():
    """Get supported languages for transcription and translation"""
    return {
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "whisper_models": transcription_service.get_available_models(),
        "translation_languages": transcription_service.get_supported_translation_languages()
    }

async def process_video_transcription(
    job_id: str,
    video_file: UploadFile,
    language: str,
    model_size: str,
    translate_to: Optional[str],
    quality_evaluation: bool,
    start_time: float
):
    """Procesa video completo: audio -> transcripción -> SRT"""
    try:
        logging.info(f"Processing job {job_id}")
        
        # 1. Guardar archivo subido
        video_path = await file_handler.save_uploaded_file(video_file, job_id)
        
        # 2. Obtener información del video
        video_info = audio_extractor.get_video_info(video_path)
        job_storage[job_id].video_info = video_info
        
        # 3. Extraer audio
        audio_path = file_handler.generate_audio_path(job_id)
        await audio_extractor.extract_audio(video_path, audio_path)
        
        # 4. Transcribir audio (con traducción integrada si se solicita)
        logging.info(f"CRITICAL DEBUG - translate_to value: '{translate_to}' (type: {type(translate_to)})")
        logging.info(f"CRITICAL DEBUG - translate_to is None: {translate_to is None}")
        logging.info(f"CRITICAL DEBUG - translate_to == 'None': {translate_to == 'None'}")
        
        # FORZAR TRADUCCIÓN PARA PRUEBA
        if translate_to in [None, 'None', '']:
            logging.warning("FORCING translate_to=es for testing")
            translate_to = 'es'
        
        logging.info(f"Calling transcription service with translate_to: {translate_to}")
        transcription_result = await transcription_service.transcribe_audio(
            audio_path=audio_path, 
            language=language, 
            model_size=model_size, 
            translate_to=translate_to
        )
        
        job_storage[job_id].transcription_text = transcription_result['text']
        job_storage[job_id].detected_language = transcription_result['language']
        
        # Debug logging para verificar datos de traducción
        logging.info(f"Transcription result keys: {list(transcription_result.keys())}")
        logging.info(f"Has translation: {transcription_result.get('translation') is not None}")
        logging.info(f"Has translation_segments: {transcription_result.get('translation_segments') is not None}")
        
        # 5. Usar segmentos (originales o traducidos)
        segments = transcription_result['segments']
        
        # Si hay traducción disponible, usar esos segmentos para el SRT
        if transcription_result.get('translation_segments'):
            logging.info("Using translated segments for SRT")
            segments = transcription_result['translation_segments']
            # También actualizar el texto principal si hay traducción
            if transcription_result.get('translation'):
                logging.info("Updating main text with translation")
                job_storage[job_id].transcription_text = transcription_result['translation']
        
        # ASEGURAR que siempre se genere el archivo SRT
        logging.info(f"Generating SRT with {len(segments)} segments")
        
        # 6. Generar SRT
        srt_path = file_handler.generate_srt_path(job_id)
        
        # Asegurar que existe el directorio SRT
        import os
        os.makedirs(os.path.dirname(srt_path), exist_ok=True)
        
        # Generar archivo SRT con segmentos (traducidos o originales)
        subtitle_generator.generate_srt(segments, srt_path)
        
        # Verificar que el archivo se creó
        if os.path.exists(srt_path):
            logging.info(f"SRT file successfully created: {srt_path}")
        else:
            logging.error(f"Failed to create SRT file: {srt_path}")
            srt_path = None
        
        # 7. Evaluar calidad
        quality_report = None
        if quality_evaluation:
            quality_report = quality_evaluator.evaluate_transcription(
                transcription_result, video_info['duration']
            )
        
        # 8. Actualizar resultado
        processing_time = time.time() - start_time
        
        job_storage[job_id].status = "completed"
        job_storage[job_id].srt_file_path = srt_path
        job_storage[job_id].quality_report = quality_report
        job_storage[job_id].processing_time = processing_time
        job_storage[job_id].completed_at = datetime.now()
        
        # 9. Limpiar archivos temporales (mantener SRT)
        file_handler.cleanup_job_files(job_id)
        
        logging.info(f"Job {job_id} completed successfully in {processing_time:.2f}s")
        
    except Exception as e:
        logging.error(f"Job {job_id} failed: {e}")
        job_storage[job_id].status = "failed"
        
        # Limpiar archivos en caso de error
        file_handler.cleanup_job_files(job_id)
