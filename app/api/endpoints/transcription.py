import os
import time
import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks, Request, Body
from fastapi.responses import FileResponse

from app.models.schemas import (
    TranscriptionRequest, TranscriptionResult, ErrorResponse,
    URLTranscriptionRequest, URLValidationResult, VideoUrlInfo,
    AudioTranscriptionRequest
)
from app.utils.validators import FileValidator
from app.utils.file_handler import FileHandler
from app.services.audio_extractor import AudioExtractor
from app.services.transcription_service import TranscriptionService
from app.services.subtitle_generator import SubtitleGenerator
from app.services.quality_evaluator import QualityEvaluator
from app.services.translation_service import TranslationService
from app.services.video_downloader import VideoDownloaderService
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
video_downloader = VideoDownloaderService()

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
    Transcribe video and optionally translate to target language
    
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
        
        # Validar parámetros de traducción
        if translate_to is not None and translate_to.strip() != "" and translate_to not in settings.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported translation target: {translate_to}. Supported: {list(settings.SUPPORTED_LANGUAGES.keys())}"
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

@router.get("/transcribe/{job_id}/status")
async def get_transcription_status(
    job_id: str,
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """Get transcription job status with simplified response"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    # Crear respuesta limpia y simplificada
    simplified_response = {
        "job_id": job.job_id,
        "status": job.status,
        "video_info": job.video_info,
        "transcription_text": job.transcription_text,
        "translated_text": job.translated_text,
        "detected_language": job.detected_language,
        "processing_time": job.processing_time,
        "created_at": job.created_at,
        "completed_at": job.completed_at
    }
    
    # Solo incluir información de segmentos si está completado y es necesario
    if job.status == "completed":
        # Incluir solo un resumen de segmentos
        if job.segments:
            simplified_response["segments_count"] = len(job.segments)
            simplified_response["segments_preview"] = job.segments[:3]  # Solo los primeros 3
        
        if job.translation_segments:
            simplified_response["translation_segments_count"] = len(job.translation_segments) 
            simplified_response["translation_segments_preview"] = job.translation_segments[:3]  # Solo los primeros 3
        
        if job.quality_report:
            simplified_response["quality_score"] = job.quality_report.overall_score
            simplified_response["quality_level"] = job.quality_report.quality_level
        
        simplified_response["srt_available"] = job.srt_file_path is not None and os.path.exists(job.srt_file_path) if job.srt_file_path else False
    
    return simplified_response

@router.get("/transcribe/{job_id}/details", response_model=TranscriptionResult)
async def get_transcription_details(
    job_id: str,
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """Get complete transcription job details with all segments"""
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
    """Get supported languages for transcription and translation with detailed mapping"""
    
    # Obtener idiomas de transcripción disponibles
    transcription_languages = settings.SUPPORTED_LANGUAGES
    
    # Obtener idiomas de traducción disponibles
    translation_languages = transcription_service.get_supported_translation_languages()
    
    # Obtener combinaciones de traducción disponibles
    available_translation_pairs = transcription_service.get_available_translation_pairs()
    
    return {
        "transcription_languages": transcription_languages,
        "translation_languages": translation_languages,
        "whisper_models": transcription_service.get_available_models(),
        "available_translation_pairs": available_translation_pairs,
        "notes": {
            "transcription": "All listed languages can be used for transcription with Whisper",
            "translation": "Translation pairs show source -> target language combinations with available models",
            "fallback": "If no model is available, a basic dictionary-based fallback will be used"
        }
    }

@router.get("/translation/providers")
async def get_translation_providers(
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """Get status and capabilities of translation providers"""
    try:
        from app.services.advanced_translation_service import AdvancedTranslationService
        
        advanced_service = AdvancedTranslationService()
        
        return {
            "providers": advanced_service.get_provider_status(),
            "supported_languages": advanced_service.get_supported_languages(),
            "priority_order": [provider.value for provider in advanced_service.provider_priority],
            "recommendations": {
                "highest_quality": "deepl",
                "most_languages": "microsoft",
                "privacy_focused": "libretranslate",
                "fallback": "google"
            }
        }
    except ImportError as e:
        logging.warning(f"Missing dependency for translation providers: {e}")
        return {
            "error": "Translation providers unavailable",
            "detail": f"Missing required dependency: {str(e)}. Please install with: pip install aiohttp",
            "fallback_available": True,
            "message": "Basic translation functionality is still available through the main transcription endpoints"
        }
    except Exception as e:
        logging.error(f"Error getting translation providers: {e}")
        return {
            "error": "Failed to get translation providers",
            "detail": str(e),
            "fallback_available": True,
            "message": "Basic translation functionality is still available through the main transcription endpoints"
        }

@router.post("/transcribe-url/validate")
async def validate_video_url(
    request: URLTranscriptionRequest = Body(...),
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """Validate video URL and get video information"""
    try:
        validation_result = video_downloader.validate_url_accessibility(request.url)
        
        if not validation_result['valid']:
            return URLValidationResult(
                valid=False,
                accessible=False,
                errors=validation_result['errors'],
                warnings=validation_result['warnings']
            )
        
        # Crear objeto VideoUrlInfo
        video_info = VideoUrlInfo(
            title=validation_result['title'],
            duration=validation_result['duration'],
            uploader=validation_result.get('uploader', 'Unknown'),
            upload_date=validation_result.get('upload_date', 'Unknown'),
            description=validation_result.get('description', ''),
            thumbnail=validation_result.get('thumbnail'),
            webpage_url=validation_result.get('webpage_url', request.url),
            extractor=validation_result['extractor'],
            has_audio=validation_result['has_audio'],
            has_video=validation_result['has_video']
        )
        
        return URLValidationResult(
            valid=True,
            accessible=True,
            video_info=video_info,
            warnings=validation_result['warnings'],
            errors=validation_result['errors']
        )
        
    except Exception as e:
        logging.error(f"Error validating URL {request.url}: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "URL Validation Failed",
                "detail": str(e),
                "error_code": "URL_VALIDATION_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )

@router.post("/transcribe-url", response_model=TranscriptionResult)
async def transcribe_video_from_url(
    request: Request,
    background_tasks: BackgroundTasks,
    transcription_request: URLTranscriptionRequest = Body(...),
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """
    Transcribe video from URL and optionally translate to target language
    
    - **url**: Video URL (YouTube, Vimeo, etc.)
    - **language**: Language code or 'auto' for detection
    - **model_size**: Whisper model size (tiny, base, small, medium, large)
    - **translate_to**: Target language for translation (optional)
    - **quality_evaluation**: Enable quality evaluation (default: true)
    - **video_quality**: Download quality (low, medium, best)
    """
    job_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Verificar rate limit
        await security_manager.check_rate_limit(request)
        
        # Validar parámetros
        if transcription_request.language not in settings.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {transcription_request.language}. Supported: {list(settings.SUPPORTED_LANGUAGES.keys())}"
            )
        
        if transcription_request.model_size not in transcription_service.get_available_models():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model size: {transcription_request.model_size}. Available: {transcription_service.get_available_models()}"
            )
        
        # Validar parámetros de traducción
        if (transcription_request.translate_to is not None and 
            transcription_request.translate_to.strip() and
            transcription_request.translate_to.strip() != "" and 
            transcription_request.translate_to not in settings.SUPPORTED_LANGUAGES):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported translation target: {transcription_request.translate_to}. Supported: {list(settings.SUPPORTED_LANGUAGES.keys())}"
            )
        
        # Validar URL
        validation_result = video_downloader.validate_url_accessibility(transcription_request.url)
        if not validation_result['valid']:
            error_details = {
                "error": "URL Validation Failed",
                "detail": "The provided URL is not accessible or valid",
                "validation_errors": validation_result['errors'],
                "warnings": validation_result['warnings'],
                "timestamp": datetime.now().isoformat()
            }
            raise HTTPException(status_code=422, detail=error_details)
        
        # Log warnings if any
        if validation_result['warnings']:
            logging.warning(f"URL validation warnings for {transcription_request.url}: {validation_result['warnings']}")
        
        # Crear entrada de trabajo para URL
        job_storage[job_id] = TranscriptionResult(
            job_id=job_id,
            status="processing",
            source_type="url",
            source_url=transcription_request.url,
            created_at=datetime.now()
        )
        
        logging.info(f"Starting URL transcription job {job_id} for user {user_info.get('name')} - URL: {transcription_request.url}")
        
        # Procesar en background
        background_tasks.add_task(
            process_url_video_transcription,
            job_id=job_id,
            transcription_request=transcription_request,
            start_time=start_time
        )
        
        return job_storage[job_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error starting URL transcription job: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "detail": f"Failed to start URL transcription: {str(e)}",
                "error_code": "PROCESSING_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )

@router.get("/video-downloader/supported-sites")
async def get_supported_sites():
    """Get list of supported video sites"""
    return {
        "supported_sites": video_downloader.get_supported_sites(),
        "total_count": len(video_downloader.get_supported_sites()),
        "note": "This is a selection of popular sites. yt-dlp supports many more sites.",
        "documentation": "https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md"
    }

@router.post("/transcribe-audio", response_model=TranscriptionResult)
async def transcribe_audio_file(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form("auto"),
    model_size: Optional[str] = Form("base"),
    translate_to: Optional[str] = Form(None),
    quality_evaluation: Optional[bool] = Form(True),
    user_info: dict = Depends(security_manager.verify_api_key)
):
    """
    Transcribe audio file and optionally translate to target language
    
    - **audio_file**: Audio file (.mp3, .wav, .flac, .aac, .ogg, .m4a, .wma)
    - **language**: Language code or 'auto' for detection
    - **model_size**: Whisper model size (tiny, base, small, medium, large)
    - **translate_to**: Target language for translation (optional)
    - **quality_evaluation**: Enable quality evaluation (default: true)
    """
    job_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log de parámetros recibidos
    logging.info(f"Audio endpoint received parameters - language: {language}, model_size: {model_size}, translate_to: {translate_to}")
    
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
        
        # Validar parámetros de traducción
        if translate_to is not None and translate_to.strip() != "" and translate_to not in settings.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported translation target: {translate_to}. Supported: {list(settings.SUPPORTED_LANGUAGES.keys())}"
            )
        
        # Validar archivo de audio con respuesta detallada
        validation_result = await file_validator.validate_audio_file(audio_file, settings)
        
        if not validation_result['valid']:
            # Combinar todos los errores en una respuesta estructurada
            error_details = {
                "error": "Audio File Validation Failed",
                "detail": "One or more validation errors occurred",
                "validation_errors": validation_result['errors'],
                "timestamp": datetime.now().isoformat()
            }
            raise HTTPException(status_code=422, detail=error_details)
        
        # Log warnings if any
        if validation_result['warnings']:
            logging.warning(f"Audio validation warnings for {audio_file.filename}: {validation_result['warnings']}")
        
        # Crear entrada de trabajo
        job_storage[job_id] = TranscriptionResult(
            job_id=job_id,
            status="processing",
            source_type="audio",
            created_at=datetime.now()
        )
        
        logging.info(f"Starting audio transcription job {job_id} for user {user_info.get('name')}")
        
        # Procesar en background
        background_tasks.add_task(
            process_audio_transcription,
            job_id=job_id,
            audio_file=audio_file,
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
        logging.error(f"Validation error in audio transcription job: {e}")
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "Validation Error",
                "detail": str(e),
                "error_code": "VALIDATION_FAILED",
                "suggestions": ["Check file size and format", "Ensure file is a valid audio file"]
            }
        )
    except Exception as e:
        logging.error(f"Error starting audio transcription job: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal Server Error",
                "detail": f"Failed to start audio transcription: {str(e)}",
                "error_code": "PROCESSING_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )

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
        video_info_dict = audio_extractor.get_video_info(video_path)
        from app.models.schemas import VideoInfo
        video_info = VideoInfo(**video_info_dict)
        job_storage[job_id].video_info = video_info
        
        # 3. Extraer audio
        audio_path = file_handler.generate_audio_path(job_id)
        await audio_extractor.extract_audio(video_path, audio_path)
        
        # 4. Transcribir audio (con traducción integrada si se solicita)
        logging.info(f"Transcription parameters - language: {language}, model_size: {model_size}, translate_to: {translate_to}")
        
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
        logging.info(f"Original segments count: {len(transcription_result.get('segments', []))}")
        logging.info(f"Translation segments count: {len(transcription_result.get('translation_segments', []))}")
        
        # 5. Usar segmentos (originales o traducidos)
        segments = transcription_result['segments']
        use_translated = False
        
        # Si hay traducción disponible, actualizar el texto principal y usar segmentos traducidos
        if transcription_result.get('translation'):
            logging.info("Translation found. Updating main text with translation")
            job_storage[job_id].transcription_text = transcription_result['translation']
            job_storage[job_id].translated_text = transcription_result['translation']
            use_translated = True
            
        if transcription_result.get('translation_segments'):
            logging.info("Using translated segments for SRT")
            segments = transcription_result['translation_segments']
        
        # ASEGURAR que siempre se genere el archivo SRT
        logging.info(f"Generating SRT with {len(segments)} segments (use_translated: {use_translated})")
        
        # Debug: Listar primeros segmentos para verificar contenido
        for i, seg in enumerate(segments[:3]):
            logging.info(f"Segment {i}: start={seg.get('start')}, end={seg.get('end')}, text='{seg.get('text', '')[:50]}...', translation='{seg.get('translation', 'N/A')[:50]}...'")
        
        # 6. Generar SRT usando el TEXTO COMPLETO para evitar pérdida de contenido
        srt_path = file_handler.generate_srt_path(job_id)
        if not srt_path or not os.path.dirname(srt_path):
            raise ValueError(f"Invalid SRT path generated: {srt_path}")

        # Asegurar que existe el directorio SRT
        os.makedirs(os.path.dirname(srt_path), exist_ok=True)

        # GENERAR SRT CON TIEMPOS CORRECTOS DE WHISPER
        # Usar segmentos traducidos si están disponibles, sino segmentos originales
        segments_to_use = []
        
        if use_translated and transcription_result.get('translation_segments'):
            segments_to_use = transcription_result['translation_segments']
            logging.info(f"Using {len(segments_to_use)} translated segments with Whisper timing")
        else:
            segments_to_use = segments
            logging.info(f"Using {len(segments_to_use)} original segments with Whisper timing")
        
        # Verificar que los segmentos tienen los campos necesarios
        valid_segments = []
        for i, seg in enumerate(segments_to_use):
            if 'start' in seg and 'end' in seg and 'text' in seg:
                # Para segmentos traducidos, usar el texto traducido
                text_to_use = seg.get('translation', seg.get('text', ''))
                if text_to_use is None:
                    text_to_use = ''
                
                valid_segment = {
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': text_to_use.strip(),
                    'translation': text_to_use.strip() if use_translated else None
                }
                valid_segments.append(valid_segment)
                logging.debug(f"Valid segment {i}: {seg['start']:.2f}-{seg['end']:.2f}s, text: '{text_to_use[:30]}...'")
            else:
                logging.warning(f"Skipping invalid segment {i}: missing required fields")
        
        logging.info(f"Processing {len(valid_segments)} valid segments for SRT generation")
        
        # Generar SRT con los tiempos correctos de Whisper
        try:
            srt_path = subtitle_generator.generate_srt(valid_segments, srt_path, use_translated=use_translated)
            logging.info(f"SRT generated successfully with correct Whisper timing: {srt_path}")
        except Exception as srt_error:
            logging.error(f"Error generating SRT with segments: {srt_error}")
            raise HTTPException(status_code=500, detail=f"SRT generation failed: {str(srt_error)}")

        job_storage[job_id].srt_file_path = srt_path
        
        # Limpiar y convertir translation_segments para la respuesta (solo información relevante)
        if transcription_result.get('translation_segments'):
            from app.models.schemas import TranscriptionSegment
            cleaned_segments = []
            for seg in transcription_result['translation_segments'][:5]:  # Solo los primeros 5 segmentos para la respuesta
                if 'start' in seg and 'end' in seg and 'text' in seg:
                    cleaned_segment = TranscriptionSegment(
                        id=seg.get('id', 0),
                        start=seg['start'],
                        end=seg['end'],
                        text=seg.get('translation', seg.get('text', '')),  # Usar traducción si está disponible
                        confidence=seg.get('confidence', 0.8)
                    )
                    cleaned_segments.append(cleaned_segment)
            job_storage[job_id].translation_segments = cleaned_segments
        
        # Para la respuesta de status, establecer segmentos originales limpios (limitados)
        if transcription_result.get('segments'):
            from app.models.schemas import TranscriptionSegment
            cleaned_original_segments = []
            for seg in transcription_result['segments'][:5]:  # Solo los primeros 5 segmentos para la respuesta
                if 'start' in seg and 'end' in seg and 'text' in seg:
                    cleaned_segment = TranscriptionSegment(
                        id=seg.get('id', 0),
                        start=seg['start'],
                        end=seg['end'],
                        text=seg.get('text', ''),
                        confidence=seg.get('confidence', 0.8)
                    )
                    cleaned_original_segments.append(cleaned_segment)
            job_storage[job_id].segments = cleaned_original_segments
        
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
                transcription_result, video_info.duration
            )
        
        # 8. Actualizar resultado
        processing_time = time.time() - start_time
        
        job_storage[job_id].status = "completed"
        job_storage[job_id].srt_file_path = srt_path
        job_storage[job_id].quality_report = quality_report
        job_storage[job_id].processing_time = processing_time
        job_storage[job_id].completed_at = datetime.now()
        
        # Limpiar archivos temporales después de completar el trabajo
        file_handler.cleanup_job_files(job_id)
        logging.info(f"Temporary files cleaned for job {job_id}")
        
    except Exception as e:
        logging.error(f"Job {job_id} failed: {e}")
        job_storage[job_id].status = "failed"
        
        # Limpiar archivos en caso de error
        file_handler.cleanup_job_files(job_id)
        raise

async def process_url_video_transcription(
    job_id: str,
    transcription_request: URLTranscriptionRequest, 
    start_time: float
):
    """Procesa transcripción de video desde URL de manera asíncrona"""
    downloaded_file_path = None
    try:
        logging.info(f"Starting URL video processing for job {job_id}")
        
        # 1. Descargar video desde URL
        download_result = await video_downloader.download_video(
            transcription_request.url,
            quality=transcription_request.video_quality,
            job_id=job_id
        )
        
        if not download_result['success']:
            raise Exception(f"Video download failed: {download_result['error']}")
        
        downloaded_file_path = download_result['file_path']
        job_storage[job_id].source_filename = os.path.basename(downloaded_file_path)
        
        logging.info(f"Video downloaded successfully for job {job_id}: {downloaded_file_path}")
        
        # 2. Extraer audio
        audio_path = file_handler.generate_audio_path(job_id)
        await audio_extractor.extract_audio(downloaded_file_path, audio_path)
        
        logging.info(f"Audio extracted for job {job_id}: {audio_path}")
        
        # 3. Obtener información del video
        video_info_dict = audio_extractor.get_video_info(downloaded_file_path)
        job_storage[job_id].duration = video_info_dict.get('duration', 0)
        
        # 4. Transcribir
        transcription_result = await transcription_service.transcribe_audio(
            audio_path, 
            transcription_request.language, 
            transcription_request.model_size
        )
        
        # Debug: Verificar el tipo y contenido de transcription_result
        logging.info(f"Transcription result type: {type(transcription_result)}")
        
        # Registrar solo información básica para evitar problemas de encoding
        if isinstance(transcription_result, dict):
            text_preview = str(transcription_result.get('text', ''))[:100] if transcription_result.get('text') else 'No text'
            # Escapar caracteres Unicode problemáticos
            text_preview = text_preview.encode('ascii', 'ignore').decode('ascii')
            logging.info(f"Transcription text preview (first 100 chars): {text_preview}...")
            logging.info(f"Language detected: {transcription_result.get('language', 'unknown')}")
            logging.info(f"Segments count: {len(transcription_result.get('segments', []))}")
            logging.info(f"Has translation: {transcription_result.get('translation') is not None}")
        else:
            logging.info(f"Transcription result is not a dict: {type(transcription_result)}")
        
        if not transcription_result:
            raise Exception("Transcription failed")
        
        # Verificar que es un diccionario antes de acceder a sus elementos
        if not isinstance(transcription_result, dict):
            raise Exception(f"Expected dict from transcription service, got {type(transcription_result)}: {transcription_result}")
        
        job_storage[job_id].language = transcription_result.get('language', transcription_request.language)
        job_storage[job_id].model_used = transcription_request.model_size
        
        # Agregar texto de transcripción principal
        if 'text' in transcription_result:
            job_storage[job_id].transcription_text = transcription_result['text']
            logging.info("Main transcription text assigned successfully")
        else:
            logging.error(f"No 'text' key in transcription_result. Keys: {list(transcription_result.keys())}")
        
        # 5. Traducir si es necesario
        translated_text = None
        translation_segments = None
        if (transcription_request.translate_to and 
            transcription_request.translate_to.strip() and 
            transcription_request.translate_to.strip() != "" and 
            transcription_request.translate_to != transcription_result.get('language')):
            
            # Verificar que 'text' existe en transcription_result
            if 'text' not in transcription_result:
                raise Exception(f"'text' key missing from transcription_result. Available keys: {list(transcription_result.keys())}")
            
            translated_text = await translation_service.translate_text(
                transcription_result['text'], 
                source_language=transcription_result.get('language', 'auto'),
                target_language=transcription_request.translate_to
            )
            
            # Traducir segmentos si están disponibles
            if transcription_result.get('segments'):
                # Verificar que 'segments' es una lista
                if not isinstance(transcription_result['segments'], list):
                    raise Exception(f"Expected list for 'segments', got {type(transcription_result['segments'])}")
                
                translation_segments = await translation_service.translate_segments(
                    transcription_result['segments'],
                    target_language=transcription_request.translate_to,
                    source_language=transcription_result.get('language', 'auto')
                )
                transcription_result['translation_segments'] = translation_segments
            
            job_storage[job_id].translated_text = translated_text
            job_storage[job_id].target_language = transcription_request.translate_to
        
        # 6. Generar SRT
        srt_path = None
        if transcription_result.get('segments'):
            logging.info(f"Found {len(transcription_result['segments'])} segments for SRT generation")
            srt_filename = f"{job_id}.srt"
            srt_path = os.path.join(settings.SRT_TEMP_DIR, srt_filename)
            
            # Determinar qué segmentos usar para SRT
            use_translated = (translation_segments is not None and 
                            transcription_request.translate_to and 
                            transcription_request.translate_to.strip() and
                            transcription_request.translate_to.strip() != "")
            segments_to_use = translation_segments if use_translated else transcription_result['segments']
            
            logging.info(f"Using translated segments: {use_translated}")
            logging.info(f"Segments to use type: {type(segments_to_use)}")
            logging.info(f"Number of segments to use: {len(segments_to_use) if segments_to_use else 0}")
            
            # Verificar que los segmentos tienen los campos necesarios
            valid_segments = []
            for i, seg in enumerate(segments_to_use):
                if i < 3:  # Solo debug para los primeros 3 segmentos
                    seg_text = str(seg.get('text', ''))[:30] if isinstance(seg, dict) else 'Invalid segment'
                    # Escapar caracteres Unicode problemáticos para logging
                    seg_text_safe = seg_text.encode('ascii', 'ignore').decode('ascii')
                    logging.debug(f"Processing segment {i}: type={type(seg)}, text_preview='{seg_text_safe}...'")
                
                if isinstance(seg, dict) and 'start' in seg and 'end' in seg and 'text' in seg:
                    text_to_use = seg.get('translation', seg.get('text', ''))
                    if text_to_use is None:
                        text_to_use = ''
                    valid_segment = {
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': text_to_use.strip(),
                        'translation': text_to_use.strip() if use_translated else None
                    }
                    valid_segments.append(valid_segment)
                else:
                    logging.warning(f"Skipping invalid segment {i}: type={type(seg)}, missing required fields or not a dict")
            
            logging.info(f"Processing {len(valid_segments)} valid segments for SRT generation")
            
            # Generar SRT
            try:
                srt_path = subtitle_generator.generate_srt(valid_segments, srt_path, use_translated=use_translated)
                logging.info(f"SRT generated successfully: {srt_path}")
            except Exception as srt_error:
                logging.error(f"Error generating SRT: {srt_error}")
                srt_path = None
        
        job_storage[job_id].srt_file_path = srt_path
        
        # Limpiar segmentos para la respuesta
        try:
            if transcription_result.get('translation_segments'):
                logging.info("Cleaning translation_segments for response")
                from app.models.schemas import TranscriptionSegment
                cleaned_segments = []
                for i, seg in enumerate(transcription_result['translation_segments'][:5]):
                    if i < 3:  # Solo debug para los primeros 3 segmentos
                        seg_text = str(seg.get('text', ''))[:20] if isinstance(seg, dict) else 'Invalid'
                        seg_text_safe = seg_text.encode('ascii', 'ignore').decode('ascii')
                        logging.debug(f"Cleaning translation segment {i}: type={type(seg)}, text='{seg_text_safe}...'")
                    
                    if isinstance(seg, dict) and 'start' in seg and 'end' in seg and 'text' in seg:
                        cleaned_segment = TranscriptionSegment(
                            id=seg.get('id', 0),
                            start=seg['start'],
                            end=seg['end'],
                            text=seg.get('translation', seg.get('text', '')),
                            confidence=seg.get('confidence', 0.8)
                        )
                        cleaned_segments.append(cleaned_segment)
                    else:
                        logging.warning(f"Skipping invalid translation segment {i}: type={type(seg)}")
                job_storage[job_id].translation_segments = cleaned_segments
                logging.info(f"Cleaned {len(cleaned_segments)} translation segments")
            
            if transcription_result.get('segments'):
                logging.info("Cleaning original segments for response")
                from app.models.schemas import TranscriptionSegment
                cleaned_original_segments = []
                for i, seg in enumerate(transcription_result['segments'][:5]):
                    if i < 3:  # Solo debug para los primeros 3 segmentos
                        seg_text = str(seg.get('text', ''))[:20] if isinstance(seg, dict) else 'Invalid'
                        seg_text_safe = seg_text.encode('ascii', 'ignore').decode('ascii')
                        logging.debug(f"Cleaning original segment {i}: type={type(seg)}, text='{seg_text_safe}...'")
                    
                    if isinstance(seg, dict) and 'start' in seg and 'end' in seg and 'text' in seg:
                        cleaned_segment = TranscriptionSegment(
                            id=seg.get('id', 0),
                            start=seg['start'],
                            end=seg['end'],
                            text=seg.get('text', ''),
                            confidence=seg.get('confidence', 0.8)
                        )
                        cleaned_original_segments.append(cleaned_segment)
                    else:
                        logging.warning(f"Skipping invalid original segment {i}: type={type(seg)}")
                job_storage[job_id].segments = cleaned_original_segments
                logging.info(f"Cleaned {len(cleaned_original_segments)} original segments")
        except Exception as cleanup_error:
            logging.error(f"Error cleaning segments for response: {cleanup_error}")
            # Continuar sin segmentos limpios en caso de error
            job_storage[job_id].translation_segments = []
            job_storage[job_id].segments = []
        
        # 7. Evaluar calidad
        quality_report = None
        if transcription_request.quality_evaluation:
            quality_report = quality_evaluator.evaluate_transcription(
                transcription_result, video_info_dict.get('duration', 0)
            )
        
        # 8. Actualizar resultado
        processing_time = time.time() - start_time
        
        job_storage[job_id].status = "completed"
        job_storage[job_id].quality_report = quality_report
        job_storage[job_id].processing_time = processing_time
        job_storage[job_id].completed_at = datetime.now()
        
        # Limpiar archivos temporales
        file_handler.cleanup_job_files(job_id)
        
        # Limpiar archivo descargado si existe
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            try:
                os.remove(downloaded_file_path)
                logging.info(f"Downloaded file cleaned up: {downloaded_file_path}")
            except Exception as cleanup_error:
                logging.warning(f"Failed to clean up downloaded file {downloaded_file_path}: {cleanup_error}")
        
        logging.info(f"URL transcription job {job_id} completed successfully")
        
    except Exception as e:
        logging.error(f"URL job {job_id} failed: {e}")
        job_storage[job_id].status = "failed"
        job_storage[job_id].error = str(e)
        job_storage[job_id].completed_at = datetime.now()
        
        # Limpiar archivos en caso de error
        file_handler.cleanup_job_files(job_id)
        
        # Limpiar archivo descargado si existe
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            try:
                os.remove(downloaded_file_path)
                logging.info(f"Downloaded file cleaned up after error: {downloaded_file_path}")
            except Exception as cleanup_error:
                logging.warning(f"Failed to clean up downloaded file {downloaded_file_path}: {cleanup_error}")

async def process_audio_transcription(
    job_id: str,
    audio_file: UploadFile,
    language: str,
    model_size: str,
    translate_to: Optional[str],
    quality_evaluation: bool,
    start_time: float
):
    """Procesa archivo de audio: procesar -> transcripción -> SRT"""
    try:
        logging.info(f"Processing audio job {job_id}")
        
        # 1. Guardar archivo de audio subido
        audio_path = await file_handler.save_uploaded_file(audio_file, job_id)
        
        # 2. Obtener información del audio
        audio_info_dict = audio_extractor.get_audio_info(audio_path)
        from app.models.schemas import AudioInfo
        audio_info = AudioInfo(**audio_info_dict)
        job_storage[job_id].audio_info = audio_info
        
        # 3. Procesar audio para Whisper (ya está en formato correcto)
        processed_audio_path = file_handler.generate_audio_path(job_id)
        await audio_extractor.process_audio_file(audio_path, processed_audio_path)
        
        # 4. Transcribir audio
        logging.info(f"Audio transcription parameters - language: {language}, model_size: {model_size}, translate_to: {translate_to}")
        
        transcription_result = await transcription_service.transcribe_audio(
            audio_path=processed_audio_path, 
            language=language, 
            model_size=model_size
        )
        
        job_storage[job_id].transcription_text = transcription_result['text']
        job_storage[job_id].detected_language = transcription_result['language']
        
        # 5. Traducir si es necesario
        translated_text = None
        translation_segments = None
        if (translate_to and 
            translate_to.strip() and 
            translate_to.strip() != "" and 
            translate_to != transcription_result.get('language')):
            
            translated_text = await translation_service.translate_text(
                transcription_result['text'], 
                source_language=transcription_result.get('language', 'auto'),
                target_language=translate_to
            )
            
            # Traducir segmentos si están disponibles
            if transcription_result.get('segments'):
                translation_segments = await translation_service.translate_segments(
                    transcription_result['segments'],
                    target_language=translate_to,
                    source_language=transcription_result.get('language', 'auto')
                )
                transcription_result['translation_segments'] = translation_segments
            
            job_storage[job_id].translated_text = translated_text
            job_storage[job_id].target_language = translate_to
        
        # 6. Generar SRT
        srt_path = None
        if transcription_result.get('segments'):
            logging.info(f"Found {len(transcription_result['segments'])} segments for SRT generation")
            srt_filename = f"{job_id}.srt"
            srt_path = os.path.join(settings.SRT_TEMP_DIR, srt_filename)
            
            # Determinar qué segmentos usar para SRT
            use_translated = (translation_segments is not None and 
                            translate_to and 
                            translate_to.strip() and
                            translate_to.strip() != "")
            segments_to_use = translation_segments if use_translated else transcription_result['segments']
            
            # Verificar que los segmentos tienen los campos necesarios
            valid_segments = []
            for i, seg in enumerate(segments_to_use):
                if isinstance(seg, dict) and 'start' in seg and 'end' in seg and 'text' in seg:
                    text_to_use = seg.get('translation', seg.get('text', ''))
                    if text_to_use is None:
                        text_to_use = ''
                    valid_segment = {
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': text_to_use.strip(),
                        'translation': text_to_use.strip() if use_translated else None
                    }
                    valid_segments.append(valid_segment)
            
            # Generar SRT
            try:
                srt_path = subtitle_generator.generate_srt(valid_segments, srt_path, use_translated=use_translated)
                logging.info(f"SRT generated successfully: {srt_path}")
            except Exception as srt_error:
                logging.error(f"Error generating SRT: {srt_error}")
                srt_path = None
        
        job_storage[job_id].srt_file_path = srt_path
        
        # 7. Evaluar calidad
        quality_report = None
        if quality_evaluation:
            quality_report = quality_evaluator.evaluate_transcription(
                transcription_result, audio_info.duration
            )
        
        # 8. Actualizar resultado
        processing_time = time.time() - start_time
        
        job_storage[job_id].status = "completed"
        job_storage[job_id].quality_report = quality_report
        job_storage[job_id].processing_time = processing_time
        job_storage[job_id].completed_at = datetime.now()
        
        # Limpiar archivos temporales
        file_handler.cleanup_job_files(job_id)
        logging.info(f"Audio transcription job {job_id} completed successfully")
        
    except Exception as e:
        logging.error(f"Audio job {job_id} failed: {e}")
        job_storage[job_id].status = "failed"
        job_storage[job_id].error = str(e)
        job_storage[job_id].completed_at = datetime.now()
        
        # Limpiar archivos en caso de error
        file_handler.cleanup_job_files(job_id)
        raise
