import asyncio
import base64
import logging
import os
import tempfile
import time
import uuid
from collections import deque
from datetime import datetime
from threading import Lock
from typing import Dict, Optional

import numpy as np
import whisper

from app.models.schemas import RealTimeSession, RealTimeTranscriptionResponse
from app.services.translation_service import TranslationService


class RealTimeTranscriptionServiceV2:
    """Servicio mejorado para transcripción en tiempo real de audio de micrófono"""
    
    def __init__(self):
        self.active_sessions: Dict[str, RealTimeSession] = {}
        self.audio_buffers: Dict[str, deque] = {}
        self.whisper_models: Dict[str, whisper.Whisper] = {}
        self.translation_service = TranslationService()
        self.lock = Lock()
        
        # Configuración optimizada para tiempo real
        self.min_audio_duration = 0.1  # Mínima duración de audio
        self.max_audio_duration = 30.0  # Máxima duración de audio
        
        logging.info("🎤 RealTimeTranscriptionServiceV2 initialized")
    
    def _load_model(self, model_size: str = "tiny") -> whisper.Whisper:
        """Carga y cachea modelos Whisper"""
        if model_size not in self.whisper_models:
            logging.info(f"🤖 Loading Whisper model: {model_size}")
            self.whisper_models[model_size] = whisper.load_model(model_size)
            logging.info(f"✅ Whisper model {model_size} loaded successfully")
        
        return self.whisper_models[model_size]
    
    def _save_webm_file(self, audio_base64: str, session_id: str) -> str:
        """Guarda audio base64 como archivo WebM temporal"""
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time() * 1000)
        temp_path = os.path.join(temp_dir, f"realtime_{session_id}_{timestamp}.webm")
        
        try:
            audio_bytes = base64.b64decode(audio_base64)
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            file_size = os.path.getsize(temp_path)
            logging.info(f"💾 Saved WebM: {temp_path} ({file_size} bytes)")
            return temp_path
            
        except Exception as e:
            logging.error(f"❌ Error saving WebM file: {e}")
            raise ValueError(f"Cannot save audio file: {e}")
    
    async def create_session(
        self, 
        session_id: str,
        language: str = "auto",
        model_size: str = "tiny",
        translate_to: Optional[str] = None
    ) -> RealTimeSession:
        """Crea una nueva sesión de transcripción en tiempo real"""
        
        session = RealTimeSession(
            session_id=session_id,
            status="active",
            language=language,
            model_size=model_size,
            translate_to=translate_to,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        with self.lock:
            self.active_sessions[session_id] = session
            self.audio_buffers[session_id] = deque(maxlen=10)
        
        # Precargar modelo
        self._load_model(model_size)
        
        logging.info(f"✅ Created session: {session_id} (language: {language}, model: {model_size})")
        return session
    
    async def process_audio_chunk(
        self,
        session_id: str,
        chunk_id: str,
        audio_base64: str
    ) -> Optional[RealTimeTranscriptionResponse]:
        """Procesa un chunk de audio en tiempo real con enfoque simplificado"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        start_time = time.time()
        
        logging.info(f"🎵 Processing chunk {chunk_id} for session {session_id}")
        
        try:
            # Validar datos de audio
            if not audio_base64:
                logging.warning(f"⚠️ Empty audio data for chunk {chunk_id}")
                return None
            
            # Guardar archivo temporal
            temp_audio_path = self._save_webm_file(audio_base64, session_id)
            
            try:
                # Verificar archivo
                if not os.path.exists(temp_audio_path):
                    logging.error(f"❌ Temp file not found: {temp_audio_path}")
                    return None
                    
                file_size = os.path.getsize(temp_audio_path)
                if file_size == 0:
                    logging.warning(f"⚠️ Empty file: {temp_audio_path}")
                    return None
                
                logging.info(f"📁 Processing file: {temp_audio_path} ({file_size} bytes)")
                
                # Transcribir con Whisper
                model = self._load_model(session.model_size)
                
                # Configuración optimizada para tiempo real
                whisper_options = {
                    "language": None if session.language == "auto" else session.language,
                    "task": "transcribe",
                    "fp16": False,
                    "verbose": True,
                    "word_timestamps": False,
                    "condition_on_previous_text": False,
                    "no_speech_threshold": 0.4,  # Más permisivo
                    "logprob_threshold": -0.8,   # Más permisivo
                    "temperature": 0.0           # Determinístico
                }
                
                logging.info(f"🤖 Starting Whisper transcription with options: {whisper_options}")
                result = model.transcribe(temp_audio_path, **whisper_options)
                
                # Extraer resultados
                transcription = result.get("text", "").strip()
                detected_language = result.get("language", session.language)
                segments = result.get("segments", [])
                
                logging.info(f"📝 Whisper results:")
                logging.info(f"   - Text: '{transcription}'")
                logging.info(f"   - Language: {detected_language}")
                logging.info(f"   - Segments: {len(segments)}")
                
                # Si no hay texto principal, intentar extraer de segmentos
                if not transcription and segments:
                    segment_texts = []
                    for segment in segments:
                        text = segment.get("text", "").strip()
                        if text:
                            segment_texts.append(text)
                            logging.info(f"   - Segment text: '{text}'")
                    
                    if segment_texts:
                        transcription = " ".join(segment_texts)
                        logging.info(f"📝 Extracted from segments: '{transcription}'")
                
                # Calcular duración estimada
                duration = 2.0  # Duración estimada de chunk
                
                # Traducir si es necesario
                translation = None
                if session.translate_to and transcription:
                    logging.info(f"🌍 Translating to {session.translate_to}")
                    try:
                        translation = await self.translation_service.translate_text(
                            transcription,
                            source_language=detected_language,
                            target_language=session.translate_to
                        )
                        logging.info(f"🌍 Translation: '{translation}'")
                    except Exception as e:
                        logging.error(f"❌ Translation failed: {e}")
                
                # Actualizar sesión
                with self.lock:
                    session.total_chunks += 1
                    session.total_duration += duration
                    session.last_activity = datetime.now()
                    
                    # Agregar a transcripción completa
                    if transcription:
                        if session.full_transcription:
                            session.full_transcription += " " + transcription
                        else:
                            session.full_transcription = transcription
                    
                    if translation:
                        if session.full_translation:
                            session.full_translation += " " + translation
                        else:
                            session.full_translation = translation
                
                processing_time = time.time() - start_time
                
                # Crear respuesta
                response = RealTimeTranscriptionResponse(
                    session_id=session_id,
                    chunk_id=chunk_id,
                    transcription=transcription,
                    translation=translation,
                    detected_language=detected_language,
                    confidence=0.8,  # Valor por defecto
                    is_final=False,
                    processing_time=processing_time,
                    timestamp=datetime.now()
                )
                
                logging.info(f"✅ Processed chunk {chunk_id} in {processing_time:.2f}s")
                return response
                
            finally:
                # Limpiar archivo temporal
                try:
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                        logging.debug(f"🗑️ Cleaned temp file: {temp_audio_path}")
                except Exception as e:
                    logging.warning(f"⚠️ Could not clean temp file: {e}")
        
        except Exception as e:
            logging.error(f"❌ Error processing chunk {chunk_id}: {e}")
            with self.lock:
                session.status = "error"
                session.error = str(e)
            raise
    
    async def get_session(self, session_id: str) -> Optional[RealTimeSession]:
        """Obtiene información de una sesión"""
        return self.active_sessions.get(session_id)
    
    async def pause_session(self, session_id: str) -> bool:
        """Pausa una sesión"""
        if session_id not in self.active_sessions:
            return False
        
        with self.lock:
            self.active_sessions[session_id].status = "paused"
        
        logging.info(f"⏸️ Paused session: {session_id}")
        return True
    
    async def resume_session(self, session_id: str) -> bool:
        """Reanuda una sesión"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        if session.status != "paused":
            return False
        
        with self.lock:
            session.status = "active"
            session.last_activity = datetime.now()
        
        logging.info(f"▶️ Resumed session: {session_id}")
        return True
    
    async def close_session(self, session_id: str) -> Optional[RealTimeSession]:
        """Cierra una sesión y devuelve la transcripción completa"""
        if session_id not in self.active_sessions:
            return None
        
        with self.lock:
            session = self.active_sessions[session_id]
            session.status = "closed"
            session.last_activity = datetime.now()
            
            # Limpiar buffers
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
        
        logging.info(f"🔚 Closed session: {session_id} (chunks: {session.total_chunks}, duration: {session.total_duration:.2f}s)")
        return session
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Limpia sesiones expiradas"""
        current_time = datetime.now()
        expired_sessions = []
        
        with self.lock:
            for session_id, session in self.active_sessions.items():
                age = (current_time - session.last_activity).total_seconds() / 3600
                if age > max_age_hours:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                if session_id in self.audio_buffers:
                    del self.audio_buffers[session_id]
        
        if expired_sessions:
            logging.info(f"🧹 Cleaned {len(expired_sessions)} expired sessions")
