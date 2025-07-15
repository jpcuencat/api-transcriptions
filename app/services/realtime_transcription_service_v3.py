import asyncio
import base64
import logging
import os
import tempfile
import time
import io
from collections import deque
from datetime import datetime
from threading import Lock
from typing import Dict, Optional

import numpy as np
import whisper

from app.models.schemas import RealTimeSession, RealTimeTranscriptionResponse
from app.services.translation_service import TranslationService


class RealTimeTranscriptionServiceV3:
    """Servicio ultra-simplificado para transcripción en tiempo real"""
    
    def __init__(self):
        self.active_sessions: Dict[str, RealTimeSession] = {}
        self.whisper_models: Dict[str, whisper.Whisper] = {}
        self.translation_service = TranslationService()
        self.lock = Lock()
        
        logging.info("[RealTime] RealTimeTranscriptionServiceV3 initialized - Ultra Simple Mode")
    
    def _load_model(self, model_size: str = "base") -> whisper.Whisper:
        """Carga y cachea modelos Whisper"""
        if model_size not in self.whisper_models:
            logging.info(f"[Whisper] Loading Whisper model: {model_size}")
            self.whisper_models[model_size] = whisper.load_model(model_size)
            logging.info(f"[Whisper] Model {model_size} loaded successfully")
        
        return self.whisper_models[model_size]
    
    def _convert_webm_to_audio_array(self, audio_base64: str) -> Optional[np.ndarray]:
        """Convierte WebM base64 a array de audio usando diferentes métodos"""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            logging.info(f"[Audio] Audio bytes length: {len(audio_bytes)}")
            
            # Crear archivo temporal WebM
            temp_dir = tempfile.gettempdir()
            temp_webm = os.path.join(temp_dir, f"temp_{int(time.time() * 1000)}.webm")
            
            with open(temp_webm, 'wb') as f:
                f.write(audio_bytes)
            
            try:
                # Método 1: Intentar con librosa (si está disponible)
                try:
                    import librosa
                    audio_data, sr = librosa.load(temp_webm, sr=16000, mono=True)
                    logging.info(f"[Librosa] Success - Length: {len(audio_data)}, SR: {sr}")
                    return audio_data.astype(np.float32)
                except Exception as e:
                    logging.warning(f"[Librosa] Failed: {e}")
                
                # Método 2: Usar pydub (más compatible)
                try:
                    from pydub import AudioSegment
                    from pydub.utils import which
                    
                    # Cargar con pydub
                    audio = AudioSegment.from_file(temp_webm, format="webm")
                    
                    # Convertir a mono y 16kHz
                    audio = audio.set_channels(1).set_frame_rate(16000)
                    
                    # Convertir a numpy array
                    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    audio_data = audio_data / (2**15)  # Normalizar
                    
                    logging.info(f"[Pydub] Success - Length: {len(audio_data)}")
                    return audio_data
                    
                except Exception as e:
                    logging.warning(f"[Pydub] Failed: {e}")
                
                # Método 3: Fallback - tratarlo como datos de audio raw
                logging.warning(f"[Fallback] Using RAW audio fallback")
                try:
                    # Intentar interpretar como PCM de 16 bits
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_np.astype(np.float32) / 32768.0
                    
                    # Limitar tamaño para evitar problemas
                    if len(audio_float) > 16000 * 30:  # Máximo 30 segundos
                        audio_float = audio_float[:16000 * 30]
                    
                    logging.info(f"[Fallback] RAW fallback - Length: {len(audio_float)}")
                    return audio_float
                    
                except Exception as e:
                    logging.error(f"[Fallback] RAW fallback failed: {e}")
                    return None
                    
            finally:
                # Limpiar archivo temporal
                try:
                    if os.path.exists(temp_webm):
                        os.remove(temp_webm)
                except:
                    pass
            
        except Exception as e:
            logging.error(f"[Audio] Audio conversion failed: {e}")
            return None
    
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
        
        # Precargar modelo
        self._load_model(model_size)
        
        logging.info(f"[Session] Created session: {session_id}")
        return session
    
    async def process_audio_chunk(
        self,
        session_id: str,
        chunk_id: str,
        audio_base64: str
    ) -> Optional[RealTimeTranscriptionResponse]:
        """Procesa un chunk de audio con máxima simplicidad"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        start_time = time.time()
        
        logging.info(f"[Processing] Processing chunk {chunk_id} for session {session_id}")
        
        try:
            # Validar datos
            if not audio_base64:
                logging.warning(f"[Processing] Empty audio data")
                return None
            
            # Convertir audio
            audio_data = self._convert_webm_to_audio_array(audio_base64)
            if audio_data is None:
                logging.error(f"[Processing] Could not convert audio")
                return None
            
            # Verificar duración mínima
            duration = len(audio_data) / 16000
            if duration < 0.1:
                logging.warning(f"[Processing] Audio too short: {duration:.2f}s")
                return None
            
            logging.info(f"[Processing] Audio duration: {duration:.2f}s")
            
            # Crear archivo temporal WAV para Whisper
            temp_dir = tempfile.gettempdir()
            temp_wav = os.path.join(temp_dir, f"whisper_{int(time.time() * 1000)}.wav")
            
            try:
                # Guardar como WAV usando scipy
                try:
                    from scipy.io.wavfile import write
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    write(temp_wav, 16000, audio_int16)
                    logging.info(f"[WAV] Saved WAV: {temp_wav}")
                except ImportError:
                    # Fallback: usar wave module
                    import wave
                    with wave.open(temp_wav, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(16000)  # 16kHz
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        wav_file.writeframes(audio_int16.tobytes())
                    logging.info(f"[WAV] Saved WAV (fallback): {temp_wav}")
                
                # Transcribir con Whisper
                model = self._load_model(session.model_size)
                
                logging.info(f"[Whisper] Starting Whisper transcription...")
                result = model.transcribe(
                    temp_wav,
                    language=None if session.language == "auto" else session.language,
                    task="transcribe",
                    fp16=False,
                    verbose=True,  # Más información para debug
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    temperature=0.2,  # Más flexible para audio real
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.4,  # Más sensible
                    prompt="Transcribe el siguiente audio en español:"  # Hint
                )
                
                transcription = result.get("text", "").strip()
                detected_language = result.get("language", session.language)
                
                logging.info(f"[Whisper] Transcription: '{transcription}'")
                logging.info(f"[Whisper] Language: {detected_language}")
                
                # Traducir si es necesario
                translation = None
                if session.translate_to and transcription:
                    try:
                        translation = await self.translation_service.translate_text(
                            transcription,
                            source_language=detected_language,
                            target_language=session.translate_to
                        )
                        logging.info(f"[Translation] Translation: '{translation}'")
                    except Exception as e:
                        logging.error(f"[Translation] Translation failed: {e}")
                
                # Actualizar sesión
                with self.lock:
                    session.total_chunks += 1
                    session.total_duration += duration
                    session.last_activity = datetime.now()
                    
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
                    confidence=0.8,
                    is_final=False,
                    processing_time=processing_time,
                    timestamp=datetime.now()
                )
                
                logging.info(f"[Processing] Chunk processed in {processing_time:.2f}s")
                return response
                
            finally:
                # Limpiar archivo temporal
                try:
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
                except:
                    pass
        
        except Exception as e:
            logging.error(f"[Processing] Error processing chunk {chunk_id}: {e}")
            import traceback
            logging.error(f"[Processing] Full traceback: {traceback.format_exc()}")
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
        
        logging.info(f"[Session] Paused session: {session_id}")
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
        
        logging.info(f"[Session] Resumed session: {session_id}")
        return True
    
    async def close_session(self, session_id: str) -> Optional[RealTimeSession]:
        """Cierra una sesión"""
        if session_id not in self.active_sessions:
            return None
        
        with self.lock:
            session = self.active_sessions[session_id]
            session.status = "closed"
            session.last_activity = datetime.now()
        
        logging.info(f"[Session] Closed session: {session_id}")
        return session
