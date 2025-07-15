import asyncio
import base64
import logging
import json
import time
import numpy as np
import webrtcvad
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import whisper
import tempfile
import os
import threading
from collections import deque

from app.core.config import settings
from app.services.translation_service import TranslationService
from app.models.schemas import RealTimeTranscriptionResponse, RealTimeSession

class RealTimeTranscriptionService:
    def __init__(self):
        self.models_cache = {}
        self.translation_service = TranslationService()
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
        self.active_sessions: Dict[str, RealTimeSession] = {}
        self.audio_buffers: Dict[str, deque] = {}
        self.lock = threading.Lock()
    
    def _load_model(self, model_size: str = "tiny"):
        """Carga modelo Whisper para tiempo real (preferiblemente tiny)"""
        if model_size not in self.models_cache:
            logging.info(f"Loading Whisper model: {model_size}")
            self.models_cache[model_size] = whisper.load_model(model_size)
        return self.models_cache[model_size]
    
    def _detect_voice_activity(self, audio_data: bytes, sample_rate: int = 16000) -> bool:
        """Detecta actividad de voz usando WebRTC VAD"""
        try:
            # WebRTC VAD requiere 16kHz, 16-bit PCM
            if sample_rate not in [8000, 16000, 32000, 48000]:
                return True  # Si no podemos validar, asumimos que hay voz
            
            # El audio debe ser en chunks de 10ms, 20ms o 30ms
            frame_duration = 20  # ms
            frame_size = int(sample_rate * frame_duration / 1000)
            
            if len(audio_data) < frame_size * 2:  # 2 bytes por sample (16-bit)
                return False
            
            # Procesar en chunks
            frames = []
            for i in range(0, len(audio_data) - frame_size * 2, frame_size * 2):
                frame = audio_data[i:i + frame_size * 2]
                frames.append(frame)
            
            # Si al menos 30% de los frames tienen voz, consideramos que hay actividad
            voice_frames = 0
            for frame in frames:
                if len(frame) == frame_size * 2:
                    try:
                        if self.vad.is_speech(frame, sample_rate):
                            voice_frames += 1
                    except:
                        continue
            
            voice_ratio = voice_frames / len(frames) if frames else 0
            return voice_ratio > 0.3
            
        except Exception as e:
            logging.warning(f"Voice activity detection failed: {e}")
            return True  # En caso de error, procesamos el audio
    
    def _audio_base64_to_numpy(self, audio_base64: str) -> np.ndarray:
        """Convierte audio base64 (WebM/Opus) a numpy array para Whisper"""
        try:
            # Decodificar base64
            audio_bytes = base64.b64decode(audio_base64)
            logging.info(f"🎵 Audio bytes length: {len(audio_bytes)}")
            
            # Crear archivo temporal directo para Whisper
            temp_dir = tempfile.gettempdir()
            temp_webm = os.path.join(temp_dir, f"temp_audio_{int(time.time())}.webm")
            
            try:
                # Guardar bytes como archivo WebM
                with open(temp_webm, 'wb') as f:
                    f.write(audio_bytes)
                
                logging.info(f"💾 Saved WebM file: {temp_webm}, size: {os.path.getsize(temp_webm)} bytes")
                
                # Intentar cargar directamente con librosa (soporta WebM)
                try:
                    import librosa
                    audio_data, sr = librosa.load(temp_webm, sr=16000, mono=True)
                    logging.info(f"🎵 Librosa loaded - Length: {len(audio_data)}, Sample rate: {sr}")
                    return audio_data.astype(np.float32)
                    
                except Exception as librosa_error:
                    logging.warning(f"⚠️ Librosa failed: {librosa_error}")
                    
                    # Fallback: usar ffmpeg
                    temp_wav = os.path.join(temp_dir, f"temp_output_{int(time.time())}.wav")
                    try:
                        import subprocess
                        result = subprocess.run([
                            'ffmpeg', '-i', temp_webm,
                            '-ar', '16000',  # Sample rate 16kHz
                            '-ac', '1',      # Mono
                            '-f', 'wav',     # Format WAV
                            '-y',            # Overwrite
                            temp_wav
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            # Leer archivo WAV convertido
                            import soundfile as sf
                            audio_data, sample_rate = sf.read(temp_wav)
                            logging.info(f"🎵 FFmpeg converted - Length: {len(audio_data)}, Sample rate: {sample_rate}")
                            
                            # Asegurar que es mono
                            if len(audio_data.shape) > 1:
                                audio_data = np.mean(audio_data, axis=1)
                            
                            return audio_data.astype(np.float32)
                        else:
                            logging.error(f"❌ FFmpeg error: {result.stderr}")
                            
                    except Exception as ffmpeg_error:
                        logging.error(f"❌ FFmpeg failed: {ffmpeg_error}")
                    finally:
                        # Limpiar archivo WAV
                        try:
                            if os.path.exists(temp_wav):
                                os.remove(temp_wav)
                        except:
                            pass
                
                # Último fallback: intentar como PCM directo
                logging.warning(f"⚠️ Using PCM fallback conversion")
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0
                logging.info(f"🎵 PCM fallback - Length: {len(audio_float)}")
                return audio_float
                
            finally:
                # Limpiar archivo WebM
                try:
                    if os.path.exists(temp_webm):
                        os.remove(temp_webm)
                except:
                    pass
            
        except Exception as e:
            logging.error(f"❌ Error converting audio: {e}")
            raise ValueError(f"Cannot convert audio data: {e}")
    
    def _save_temp_audio(self, audio_np: np.ndarray, session_id: str) -> str:
        """Guarda audio temporal para Whisper"""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"realtime_audio_{session_id}_{int(time.time())}.wav")
        
        try:
            # Whisper espera archivos de audio, así que guardamos temporalmente
            import soundfile as sf
            sf.write(temp_path, audio_np, 16000)  # 16kHz sample rate
            logging.info(f"💾 Saved temp audio file: {temp_path}, size: {os.path.getsize(temp_path)} bytes")
        except Exception as e:
            logging.error(f"❌ Error saving temp audio: {e}")
            # Fallback: usar scipy
            try:
                from scipy.io.wavfile import write
                # Convertir a int16 para scipy
                audio_int16 = (audio_np * 32767).astype(np.int16)
                write(temp_path, 16000, audio_int16)
                logging.info(f"💾 Saved temp audio (scipy fallback): {temp_path}")
            except Exception as e2:
                logging.error(f"❌ Scipy fallback failed: {e2}")
                raise ValueError(f"Cannot save audio file: {e}")
        
        return temp_path
    
    def _save_temp_audio_from_base64(self, audio_base64: str, session_id: str) -> str:
        """Guarda audio base64 directamente como archivo temporal para Whisper"""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"realtime_audio_{session_id}_{int(time.time())}.webm")
        
        try:
            # Decodificar y guardar directamente como WebM
            audio_bytes = base64.b64decode(audio_base64)
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            logging.info(f"💾 Saved temp WebM file: {temp_path}, size: {os.path.getsize(temp_path)} bytes")
            return temp_path
            
        except Exception as e:
            logging.error(f"❌ Error saving temp audio from base64: {e}")
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
            self.audio_buffers[session_id] = deque(maxlen=10)  # Buffer de últimos 10 chunks
        
        # Precargar modelo
        self._load_model(model_size)
        
        logging.info(f"Created real-time transcription session: {session_id}")
        return session
    
    async def process_audio_chunk(
        self,
        session_id: str,
        chunk_id: str,
        audio_base64: str,
        enable_vad: bool = False,  # Deshabilitamos VAD temporalmente
        min_speech_duration: float = 0.1  # Reducimos duración mínima
    ) -> Optional[RealTimeTranscriptionResponse]:
        """Procesa un chunk de audio en tiempo real"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        start_time = time.time()
        
        logging.info(f"🎵 Processing audio chunk {chunk_id} for session {session_id}")
        
        try:
            # Convertir audio
            audio_np = self._audio_base64_to_numpy(audio_base64)
            
            # Verificar duración mínima
            duration = len(audio_np) / 16000  # Asumiendo 16kHz
            logging.info(f"📏 Audio duration: {duration:.2f}s (min required: {min_speech_duration}s)")
            
            if duration < min_speech_duration:
                logging.warning(f"⚠️ Audio chunk too short: {duration:.2f}s")
                return None
            
            # Detectar actividad de voz si está habilitado
            if enable_vad:
                audio_bytes = base64.b64decode(audio_base64)
                voice_detected = self._detect_voice_activity(audio_bytes)
                logging.info(f"🎤 Voice activity detected: {voice_detected}")
                if not voice_detected:
                    logging.warning(f"⚠️ No voice activity detected in chunk {chunk_id}")
                    return None
            
            # Agregar al buffer de la sesión
            with self.lock:
                self.audio_buffers[session_id].append(audio_np)
            
            # Guardar audio temporal - ahora directamente del base64
            temp_audio_path = self._save_temp_audio_from_base64(audio_base64, session_id)
            
            try:
                # Transcribir con Whisper
                model = self._load_model(session.model_size)
                logging.info(f"🤖 Using Whisper model: {session.model_size}")
                
                # Verificar que el archivo existe y tiene contenido
                if not os.path.exists(temp_audio_path):
                    logging.error(f"❌ Temp audio file not found: {temp_audio_path}")
                    return None
                    
                file_size = os.path.getsize(temp_audio_path)
                logging.info(f"📁 Temp audio file size: {file_size} bytes")
                
                if file_size == 0:
                    logging.warning(f"⚠️ Empty audio file: {temp_audio_path}")
                    return None
                
                # Para tiempo real, usamos opciones optimizadas
                logging.info(f"🎤 Starting Whisper transcription...")
                result = model.transcribe(
                    temp_audio_path,
                    language=None if session.language == "auto" else session.language,
                    task="transcribe",
                    fp16=False,  # Más estable para tiempo real
                    verbose=True,  # Activamos verbose para más info
                    word_timestamps=False,  # Desactivamos timestamps para velocidad
                    condition_on_previous_text=False  # No usar contexto previo
                )
                
                transcription = result["text"].strip()
                detected_language = result.get("language", session.language)
                confidence = result.get("confidence", 0.0)
                
                logging.info(f"📝 Whisper result:")
                logging.info(f"   - Text: '{transcription}'")
                logging.info(f"   - Language: {detected_language}")
                logging.info(f"   - Confidence: {confidence}")
                logging.info(f"   - Full result keys: {list(result.keys())}")
                
                # Si no hay transcripción pero hay segments, extraer de ahí
                if not transcription and "segments" in result:
                    segments_text = []
                    for segment in result["segments"]:
                        if "text" in segment and segment["text"].strip():
                            segments_text.append(segment["text"].strip())
                    if segments_text:
                        transcription = " ".join(segments_text)
                        logging.info(f"📝 Extracted from segments: '{transcription}'")
                
                # Si aún no hay transcripción, devolver respuesta vacía pero informativa
                if not transcription:
                    logging.warning(f"⚠️ Empty transcription for chunk {chunk_id}")
                    # Aun así, devolvemos una respuesta para mostrar que se procesó
                    return RealTimeTranscriptionResponse(
                        session_id=session_id,
                        chunk_id=chunk_id,
                        transcription="",  # Transcripción vacía
                        translation=None,
                        detected_language=detected_language,
                        confidence=confidence,
                        is_final=False,
                        processing_time=time.time() - start_time,
                        timestamp=datetime.now()
                    )
                
                # Traducir si es necesario
                translation = None
                if session.translate_to and transcription:
                    logging.info(f"🌍 Translating from {detected_language} to {session.translate_to}")
                    translation = await self.translation_service.translate_text(
                        transcription,
                        source_language=detected_language,
                        target_language=session.translate_to
                    )
                    logging.info(f"🌍 Translation result: '{translation}'")
                
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
                    confidence=None,  # Whisper no proporciona confidence fácilmente
                    is_final=False,  # En tiempo real, siempre es provisional
                    processing_time=processing_time,
                    timestamp=datetime.now()
                )
                
                return response
                
            finally:
                # Limpiar archivo temporal
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
        
        except Exception as e:
            logging.error(f"Error processing audio chunk {chunk_id}: {e}")
            with self.lock:
                session.status = "error"
                session.error = str(e)
            raise
    
    async def get_session(self, session_id: str) -> Optional[RealTimeSession]:
        """Obtiene información de una sesión"""
        return self.active_sessions.get(session_id)
    
    async def close_session(self, session_id: str) -> Optional[RealTimeSession]:
        """Cierra una sesión de transcripción"""
        if session_id in self.active_sessions:
            with self.lock:
                session = self.active_sessions[session_id]
                session.status = "completed"
                session.last_activity = datetime.now()
                
                # Limpiar buffer
                if session_id in self.audio_buffers:
                    del self.audio_buffers[session_id]
                
                return session
        return None
    
    async def pause_session(self, session_id: str) -> bool:
        """Pausa una sesión"""
        if session_id in self.active_sessions:
            with self.lock:
                self.active_sessions[session_id].status = "paused"
                return True
        return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Reanuda una sesión pausada"""
        if session_id in self.active_sessions:
            with self.lock:
                session = self.active_sessions[session_id]
                if session.status == "paused":
                    session.status = "active"
                    session.last_activity = datetime.now()
                    return True
        return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Limpia sesiones antiguas"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        with self.lock:
            for session_id, session in self.active_sessions.items():
                age = (current_time - session.last_activity).total_seconds() / 3600
                if age > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
                if session_id in self.audio_buffers:
                    del self.audio_buffers[session_id]
                logging.info(f"Cleaned up old session: {session_id}")
