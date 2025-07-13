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
        "translation_text": job.translation_text,
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
    except Exception as e:
        logging.error(f"Error getting translation providers: {e}")
        return {
            "error": "Failed to get translation providers",
            "detail": str(e)
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
            job_storage[job_id].translation_text = transcription_result['translation']
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
