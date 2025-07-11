import whisper
import torch
import logging
import os
from typing import Dict, List, Optional
from langdetect import detect
from app.core.config import settings

class TranscriptionService:
    def __init__(self):
        self.models = {}
        self.cache_dir = settings.WHISPER_CACHE_DIR
        
        # Crear directorio de cache si no existe
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_model(self, model_size: str = "base") -> whisper.Whisper:
        """Carga modelo Whisper con cache y manejo robusto de errores"""
        if model_size not in self.models:
            logging.info(f"Loading Whisper model: {model_size}")
            
            # Configurar directorio de descarga
            os.environ['WHISPER_CACHE_DIR'] = self.cache_dir
            
            try:
                # Intentar cargar con configuración estable
                self.models[model_size] = whisper.load_model(
                    model_size, 
                    download_root=self.cache_dir,
                    device="cpu"  # Forzar CPU para mayor estabilidad
                )
                logging.info(f"Model {model_size} loaded successfully on CPU")
                
                # Verificar que el modelo funciona
                test_audio = torch.zeros(16000)  # 1 segundo de silencio
                with torch.no_grad():
                    result = self.models[model_size].transcribe(
                        test_audio.numpy(), 
                        fp16=False, 
                        verbose=False,
                        word_timestamps=False
                    )
                logging.info("Model verification successful")
                
            except Exception as e:
                logging.error(f"Error loading model {model_size}: {e}")
                
                # Intentar limpieza y recarga
                if model_size in self.models:
                    del self.models[model_size]
                
                # Reintento con configuración minimal
                try:
                    logging.info("Retrying model load with minimal configuration...")
                    self.models[model_size] = whisper.load_model(model_size)
                    logging.info(f"Model {model_size} loaded on retry")
                except Exception as retry_error:
                    logging.error(f"Model reload failed: {retry_error}")
                    raise Exception(f"Failed to load Whisper model after retry: {retry_error}")
                
        return self.models[model_size]
    
    async def transcribe_audio(self,
                              audio_path: str,
                              language: str = 'auto',
                              model_size: str = 'base',
                              task: str = 'transcribe') -> Dict:
        """Transcribe audio usando Whisper"""
        try:
            logging.info(f"Starting transcription: {audio_path}")
            logging.info(f"Parameters - Language: {language}, Model: {model_size}, Task: {task}")
            
            model = self.load_model(model_size)
            
            # Verificar que el archivo de audio existe
            if not os.path.exists(audio_path):
                raise Exception(f"Audio file not found: {audio_path}")
            
            # Verificar tamaño del archivo
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise Exception(f"Audio file is empty: {audio_path}")
            
            logging.info(f"Audio file verified: {audio_path} ({file_size} bytes)")
            
            # Configuración optimizada y robusta
            options = {
                'task': task,  # 'transcribe' o 'translate'
                'language': None if language == 'auto' else language,
                'fp16': False,  # Desactivar FP16 para mayor estabilidad
                'verbose': False,
                'word_timestamps': False,  # Desactivar para mayor estabilidad
                'condition_on_previous_text': False,  # Mejor para audio largo
                'temperature': 0.0,  # Determinístico
                'compression_ratio_threshold': 2.4,
                'logprob_threshold': -1.0,
                'no_speech_threshold': 0.6,
                'beam_size': None,  # Usar default
                'best_of': None,   # Usar default
                'patience': None   # Usar default
            }
            
            # Realizar transcripción con manejo robusto
            logging.info("Running Whisper transcription...")
            
            # Método con manejo de broken pipe
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Liberar memoria antes de transcribir
                    import gc
                    gc.collect()
                    
                    # Configurar timeout y límites
                    with torch.no_grad():  # Evitar acumulación de gradientes
                        result = model.transcribe(audio_path, **options)
                    
                    logging.info("Transcription completed successfully")
                    break
                    
                except Exception as transcribe_error:
                    logging.warning(f"Transcription attempt {attempt + 1} failed: {transcribe_error}")
                    
                    if attempt < max_retries - 1:
                        logging.info("Retrying with more conservative settings...")
                        
                        # Configuración más conservadora para reintento
                        options.update({
                            'verbose': None,
                            'word_timestamps': False,
                            'beam_size': 1,  # Usar beam search más simple
                            'best_of': 1,    # Reducir candidatos
                        })
                        
                        # Forzar limpieza de memoria
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # Esperar un momento
                        import time
                        time.sleep(2)
                    else:
                        # Si falla todos los intentos, re-raise el error
                        raise transcribe_error
            
            # Detectar idioma si es auto
            detected_language = result.get('language', 'unknown')
            if language == 'auto':
                logging.info(f"Detected language: {detected_language}")
            
            # Calcular confidence score
            confidence = self._calculate_confidence(result)
            
            transcription_result = {
                'text': result['text'].strip(),
                'language': detected_language,
                'segments': result.get('segments', []),
                'duration': result.get('duration', 0),
                'confidence': confidence
            }
            
            logging.info(f"Transcription completed. Confidence: {confidence:.2f}")
            return transcription_result
            
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            raise Exception(f"Transcription failed: {e}")
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calcula confianza promedio de la transcripción"""
        if 'segments' not in result or not result['segments']:
            return 0.0
        
        confidences = []
        for segment in result['segments']:
            # Whisper no siempre incluye word-level probabilities
            # Usar avg_logprob del segmento como aproximación
            if 'avg_logprob' in segment:
                # Convertir log probability a probability
                prob = min(1.0, max(0.0, torch.exp(torch.tensor(segment['avg_logprob'])).item()))
                confidences.append(prob)
            elif 'words' in segment:
                # Si hay word-level timestamps, usar esas probabilidades
                for word in segment['words']:
                    if 'probability' in word:
                        confidences.append(word['probability'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def get_available_models(self) -> List[str]:
        """Retorna modelos Whisper disponibles"""
        return ["tiny", "base", "small", "medium", "large"]
