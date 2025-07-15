import asyncio
import base64
import logging
import os
import tempfile
import time
import io
import struct
from collections import deque
from datetime import datetime
from threading import Lock
from typing import Dict, Optional

import numpy as np
import whisper

from app.models.schemas import RealTimeSession, RealTimeTranscriptionResponse
from app.services.translation_service import TranslationService


class RealTimeTranscriptionServiceV4:
    """Servicio simplificado con mejor manejo de audio WebRTC"""
    
    def __init__(self):
        self.active_sessions: Dict[str, RealTimeSession] = {}
        self.whisper_models: Dict[str, whisper.Whisper] = {}
        self.translation_service = TranslationService()
        self.lock = Lock()
        
        logging.info("[RealTime] RealTimeTranscriptionServiceV4 initialized - Better Audio Handling")
    
    def _load_model(self, model_size: str = "base") -> whisper.Whisper:
        """Carga y cachea modelos Whisper"""
        if model_size not in self.whisper_models:
            logging.info(f"[Whisper] Loading Whisper model: {model_size}")
            self.whisper_models[model_size] = whisper.load_model(model_size)
            logging.info(f"[Whisper] Model {model_size} loaded successfully")
        
        return self.whisper_models[model_size]
    
    def _create_wav_from_raw_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Crea un archivo WAV directamente desde datos de audio raw"""
        try:
            # Crear archivo temporal WAV
            temp_dir = tempfile.gettempdir()
            temp_wav = os.path.join(temp_dir, f"whisper_raw_{int(time.time() * 1000)}.wav")
            
            # Intentar diferentes interpretaciones de los datos
            audio_arrays = []
            
            # Opción 1: Interpretar como float32 little-endian
            if len(audio_data) % 4 == 0:
                try:
                    audio_float = np.frombuffer(audio_data, dtype=np.float32)
                    if np.max(np.abs(audio_float)) <= 1.0:  # Verificar rango válido
                        audio_arrays.append(("float32", audio_float))
                        logging.info(f"[Audio] Float32 interpretation: {len(audio_float)} samples")
                except:
                    pass
            
            # Opción 2: Interpretar como int16 little-endian
            if len(audio_data) % 2 == 0:
                try:
                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32767.0
                    audio_arrays.append(("int16", audio_float))
                    logging.info(f"[Audio] Int16 interpretation: {len(audio_int16)} samples")
                except:
                    pass
            
            # Opción 3: Interpretar como unsigned int8
            try:
                audio_uint8 = np.frombuffer(audio_data, dtype=np.uint8)
                audio_float = (audio_uint8.astype(np.float32) - 128.0) / 128.0
                audio_arrays.append(("uint8", audio_float))
                logging.info(f"[Audio] Uint8 interpretation: {len(audio_uint8)} samples")
            except:
                pass
            
            # Usar la primera interpretación válida
            best_audio = None
            best_format = None
            
            for format_name, audio_array in audio_arrays:
                # Verificar que la duración sea razonable (entre 0.1 y 10 segundos)
                duration = len(audio_array) / sample_rate
                if 0.1 <= duration <= 10.0:
                    best_audio = audio_array
                    best_format = format_name
                    logging.info(f"[Audio] Using {format_name} format, duration: {duration:.2f}s")
                    break
            
            if best_audio is None:
                logging.error("[Audio] No valid audio interpretation found")
                return None
            
            # Normalizar audio si es necesario
            if np.max(np.abs(best_audio)) > 1.0:
                best_audio = best_audio / np.max(np.abs(best_audio))
                logging.info("[Audio] Audio normalized to [-1, 1] range")
            
            # Crear archivo WAV usando wave module
            import wave
            with wave.open(temp_wav, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)  # Sample rate
                
                # Convertir a int16
                audio_int16 = (best_audio * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            
            logging.info(f"[WAV] Created WAV file: {temp_wav}")
            return temp_wav
            
        except Exception as e:
            logging.error(f"[Audio] Error creating WAV from raw audio: {e}")
            return None
    
    def _convert_webm_to_audio_array(self, audio_base64: str) -> Optional[str]:
        """Convierte WebM base64 a archivo WAV para Whisper"""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            logging.info(f"[Audio] Audio bytes length: {len(audio_bytes)}")
            
            # Método 1: Intentar usar pydub con mejor configuración
            try:
                from pydub import AudioSegment
                
                # Crear archivo temporal WebM
                temp_dir = tempfile.gettempdir()
                temp_webm = os.path.join(temp_dir, f"temp_{int(time.time() * 1000)}.webm")
                
                with open(temp_webm, 'wb') as f:
                    f.write(audio_bytes)
                
                try:
                    # Intentar cargar como WebM/Opus
                    audio = AudioSegment.from_file(temp_webm)
                    
                    # Convertir a mono y 16kHz
                    audio = audio.set_channels(1).set_frame_rate(16000)
                    
                    # Crear archivo WAV temporal
                    temp_wav = os.path.join(temp_dir, f"whisper_pydub_{int(time.time() * 1000)}.wav")
                    audio.export(temp_wav, format="wav")
                    
                    logging.info(f"[Pydub] Success - Created WAV: {temp_wav}")
                    
                    # Limpiar archivo WebM temporal
                    try:
                        os.remove(temp_webm)
                    except:
                        pass
                    
                    return temp_wav
                    
                except Exception as e:
                    logging.warning(f"[Pydub] Failed to process WebM: {e}")
                    # Limpiar archivo WebM temporal
                    try:
                        os.remove(temp_webm)
                    except:
                        pass
                
            except Exception as e:
                logging.warning(f"[Pydub] Import or processing failed: {e}")
            
            # Método 2: Tratarlo como datos de audio raw
            logging.info("[Fallback] Trying raw audio interpretation")
            wav_file = self._create_wav_from_raw_audio(audio_bytes)
            if wav_file:
                return wav_file
            
            logging.error("[Audio] All conversion methods failed")
            return None
            
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
        """Procesa un chunk de audio con mejor manejo de formatos"""
        
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
            
            # Convertir audio a WAV
            wav_file = self._convert_webm_to_audio_array(audio_base64)
            if wav_file is None:
                logging.error(f"[Processing] Could not convert audio to WAV")
                return None
            
            try:
                # Transcribir con Whisper
                model = self._load_model(session.model_size)
                
                logging.info(f"[Whisper] Starting Whisper transcription on: {wav_file}")
                result = model.transcribe(
                    wav_file,
                    language=None if session.language == "auto" else session.language,
                    task="transcribe",
                    fp16=False,
                    verbose=True,  # Más información
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    temperature=0.2,  # Más flexible para reconocer habla real
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.4,  # Más sensible para captar habla suave
                    prompt="Transcribe el siguiente audio en español:"  # Hint para español
                )
                
                transcription = result.get("text", "").strip()
                detected_language = result.get("language", session.language)
                
                logging.info(f"[Whisper] Transcription: '{transcription}'")
                logging.info(f"[Whisper] Language: {detected_language}")
                logging.info(f"[Whisper] Full result keys: {list(result.keys())}")
                
                # Si no hay transcripción, verificar el audio
                if not transcription:
                    try:
                        import wave
                        with wave.open(wav_file, 'rb') as wf:
                            frames = wf.getnframes()
                            sample_rate = wf.getframerate()
                            duration = frames / sample_rate
                            logging.info(f"[WAV] File info - Frames: {frames}, SR: {sample_rate}, Duration: {duration:.2f}s")
                    except:
                        pass
                
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
                    session.total_duration += duration if 'duration' in locals() else 0
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
                # Limpiar archivo temporal WAV
                try:
                    if wav_file and os.path.exists(wav_file):
                        os.remove(wav_file)
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
